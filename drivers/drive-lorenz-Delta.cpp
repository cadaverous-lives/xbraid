/*BHEADER**********************************************************************
 * Copyright (c) 2013, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory. Written by
 * Jacob Schroder, Rob Falgout, Tzanio Kolev, Ulrike Yang, Veselin
 * Dobrev, et al. LLNL-CODE-660355. All rights reserved.
 *
 * This file is part of XBraid. For support, post issues to the XBraid Github page.
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 ***********************************************************************EHEADER*/

//
// Example:       ex-01-pp.cpp
//
// Interface:     C++
//
// Requires:      C-language and C++ support
//
// Compile with:  make drive-lorenz-Delta
//
// Help with:     ex-01-pp -help
//
// Sample run:    mpirun -np 2 drive-lorenz-Delta
//
// Description:   solve the Lorenz system using forward Euler and Delta correction
//
//

#include <iostream>
#include <cmath>
#include <string.h>
#include <fstream>

#include "braid.hpp"
#include "drive-lorenz-Delta-lib/lorenz_lib.hpp"

// --------------------------------------------------------------------------
// User-defined routines and objects
// --------------------------------------------------------------------------

// Define BraidVector, can contain anything, and be named anything
// --> Put all time-dependent information here
class BraidVector
{
public:
   // Each vector holds the state vector at a particular time
   VEC state;
   VEC prev_c_point;
   MAT Delta;

   // Construct a BraidVector for a given vector of doubles
   BraidVector(VEC state_, VEC adjnt_, MAT Delta_) : state(state_), prev_c_point(adjnt_), Delta(Delta_) {}
   BraidVector() : state(VEC()), prev_c_point(VEC()), Delta(MAT()) {}

   // Deconstructor
   virtual ~BraidVector(){};
};

// Wrapper for BRAID's App object
// --> Put all time INDEPENDENT information here
class MyBraidApp : public BraidApp
{
protected:
   // BraidApp defines tstart, tstop, ntime and comm_t

public:
   int cfactor;   // Currently only supporting one CF for all levels
   bool useDelta;
   // Constructor
   MyBraidApp(MPI_Comm comm_t_, int rank_, double tstart_ = 0.0, double tstop_ = 1.0, int ntime_ = 100, int cfactor_ = 2, bool useDelta_ = false);

   // We will need the MPI Rank
   int rank;

   // Deconstructor
   virtual ~MyBraidApp(){};

   int IsCPoint(int i, int level);

   // Define all the Braid Wrapper routines
   // Note: braid_Vector == BraidVector*
   virtual int Step(braid_Vector u_,
                    braid_Vector ustop_,
                    braid_Vector fstop_,
                    BraidStepStatus &pstatus);

   virtual int Clone(braid_Vector u_,
                     braid_Vector *v_ptr);

   virtual int Init(double t,
                    braid_Vector *u_ptr);

   virtual int Free(braid_Vector u_);

   virtual int Sum(double alpha,
                   braid_Vector x_,
                   double beta,
                   braid_Vector y_);

   virtual int SpatialNorm(braid_Vector u_,
                           double *norm_ptr);

   virtual int BufSize(int *size_ptr,
                       BraidBufferStatus &status);

   virtual int BufPack(braid_Vector u_,
                       void *buffer,
                       BraidBufferStatus &status);

   virtual int BufUnpack(void *buffer,
                         braid_Vector *u_ptr,
                         BraidBufferStatus &status);

   virtual int Access(braid_Vector u_,
                      BraidAccessStatus &astatus);

   virtual int Residual(braid_Vector u_,
                        braid_Vector r_,
                        BraidStepStatus &pstatus);

   // Not needed in this driver
   virtual int Coarsen(braid_Vector fu_,
                       braid_Vector *cu_ptr,
                       BraidCoarsenRefStatus &status) { return 0; }

   // Not needed in this driver
   virtual int Refine(braid_Vector cu_,
                      braid_Vector *fu_ptr,
                      BraidCoarsenRefStatus &status) { return 0; }
};

// Braid App Constructor
MyBraidApp::MyBraidApp(MPI_Comm comm_t_, int rank_, double tstart_, double tstop_, int ntime_, int cfactor_, bool useDelta_)
    : BraidApp(comm_t_, tstart_, tstop_, ntime_)
{
   rank = rank_;
   cfactor = cfactor_;
   useDelta = useDelta_;
}

// Helper function to check if current point is a C point for this level
int MyBraidApp::IsCPoint(int i, int level)
{
   return ((i % intpow(cfactor, level + 1)) == 0);
}

//
int MyBraidApp::Step(braid_Vector u_,
                     braid_Vector ustop_,
                     braid_Vector fstop_,
                     BraidStepStatus &pstatus)
{

   BraidVector *u = (BraidVector *)u_;
   BraidVector *f = (BraidVector *)fstop_;

   double tstart; // current time
   double tstop;  // evolve to this time
   int level, nlevels, T_index;

   pstatus.GetTstartTstop(&tstart, &tstop);
   pstatus.GetLevel(&level);
   pstatus.GetNLevels(&nlevels);
   pstatus.GetTIndex(&T_index); // this is the index of tstart

   // no refinement
   pstatus.SetRFactor(1);

   if (!useDelta) // default behavior
   {
      u->state = euler(u->state, tstop - tstart) + f->Delta * u->state;
      return 0;
   }
   // else:

   // Compute linear tangent propagators only on fine-grids
   if (level < nlevels - 1)
   {
      if (IsCPoint(T_index, level))
      {
         // Need to store the value at the previous c-point for tau correction later
         // TODO Is there a way to get around this?
         u->prev_c_point = u->state;
         // we implicitly set u->Delta = I at each C-point
         if (level == 0) {u->Delta = euler_du(u->state, tstop - tstart);}
         else {u->Delta = euler_du(u->state, tstop - tstart) + f->Delta;}
      }
      else
      {
         if (level == 0) {u->Delta = euler_du(u->state, tstop - tstart) * u->Delta;}
         else {u->Delta = (euler_du(u->state, tstop - tstart) + f->Delta) * u->Delta;}
      }
   }
   if (level == 0) { u->state = euler(u->state, tstop - tstart);}
   else {u->state = euler(u->state, tstop - tstart) + f->Delta * u->state + f->state;}

   return 0;
}

int MyBraidApp::Residual(braid_Vector u_,
                         braid_Vector r_,
                         BraidStepStatus &pstatus)
{
   BraidVector *u = (BraidVector *)u_;
   BraidVector *r = (BraidVector *)r_;

   double tstart; // current time
   double tstop;  // evolve to this time
   int level;
   pstatus.GetTstartTstop(&tstart, &tstop);
   pstatus.GetLevel(&level);

   if (!useDelta)
   {
      r->state = u->state - euler(r->state, tstop - tstart);
      return 0;
   }
   // else:
   // -D Phi^m
   r->Delta = -euler_du(r->state, tstop - tstart) * r->Delta; 
   // u_i - [Phi^m - DPhi^m]u_{i-m}
   r->state = u->state - euler(r->state, tstop - tstart) - r->Delta * r->prev_c_point;

   return 0;
}

int MyBraidApp::Init(double t,
                     braid_Vector *u_ptr)
{
   BraidVector *u = new BraidVector();
   if (t != tstart)
   {
      // this is the Eigen "comma initialization" syntax
      u->state << 0., 0., 0.;
   }
   else
   {
      u->state << -1.8430428, -0.07036326, 23.15614636;
   }

   u->Delta.setIdentity();
   u->prev_c_point << 0., 0., 0.;

   *u_ptr = (braid_Vector)u;
   return 0;
}

int MyBraidApp::Clone(braid_Vector u_,
                      braid_Vector *v_ptr)
{
   BraidVector *u = (BraidVector *)u_;
   BraidVector *v = new BraidVector(u->state, u->prev_c_point, u->Delta);
   *v_ptr = (braid_Vector)v;

   return 0;
}

int MyBraidApp::Free(braid_Vector u_)
{
   BraidVector *u = (BraidVector *)u_;
   delete u;
   return 0;
}

int MyBraidApp::Sum(double alpha,
                    braid_Vector x_,
                    double beta,
                    braid_Vector y_)
{
   BraidVector *x = (BraidVector *)x_;
   BraidVector *y = (BraidVector *)y_;
   (y->state) = alpha * (x->state) + beta * (y->state);
   (y->Delta) = alpha * (x->Delta) + beta * (y->Delta);
   return 0;
}

int MyBraidApp::SpatialNorm(braid_Vector u_,
                            double *norm_ptr)
{
   BraidVector *u = (BraidVector *)u_;
   *norm_ptr = u->state.norm();
   return 0;
}

int MyBraidApp::BufSize(int *size_ptr,
                        BraidBufferStatus &status)
{
   if (useDelta)
   {
      *size_ptr = (2 * VECSIZE + VECSIZE * VECSIZE) * sizeof(double);
   }
   else
   {
      *size_ptr = (VECSIZE) * sizeof(double);
   }

   return 0;
}

int MyBraidApp::BufPack(braid_Vector u_,
                        void *buffer,
                        BraidBufferStatus &status)
{
   BraidVector *u = (BraidVector *)u_;
   double *dbuffer = (double *)buffer;
   status.SetSize(sizeof(double));

   for (size_t i = 0; i < VECSIZE; i++)
   {
      dbuffer[i] = (u->state[i]);
   }

   if (useDelta)
   {
      for (size_t i = 0; i < VECSIZE; i++)
      {
         dbuffer[i + VECSIZE] = (u->prev_c_point[i]);
      }
      for (size_t i = 0; i < VECSIZE * VECSIZE; i++)
      {
         dbuffer[i + 2 * VECSIZE] = (u->Delta(i));
      }
   }


   return 0;
}

int MyBraidApp::BufUnpack(void *buffer,
                          braid_Vector *u_ptr,
                          BraidBufferStatus &status)
{
   double *dbuffer = (double *)buffer;

   BraidVector *u = new BraidVector();

   for (size_t i = 0; i < VECSIZE; i++)
   {
      (u->state[i]) = dbuffer[i];
   }

   if (useDelta)
   {
      for (size_t i = 0; i < VECSIZE; i++)
      {
         (u->prev_c_point[i]) = dbuffer[i + VECSIZE];
      }
      for (size_t i = 0; i < VECSIZE * VECSIZE; i++)
      {
         (u->Delta(i)) = dbuffer[i + 2 * VECSIZE];
      }
   }
   *u_ptr = (braid_Vector)u;

   return 0;
}

int MyBraidApp::Access(braid_Vector u_,
                       BraidAccessStatus &astatus)
{
   char filename[255];
   std::ofstream file;
   BraidVector *u = (BraidVector *)u_;

   // Extract information from astatus
   int done, level, iter, index;
   double t;
   astatus.GetTILD(&t, &iter, &level, &done);
   astatus.GetTIndex(&index);
   int nt = MyBraidApp::ntime;

   // Print information to file
   if ((index == nt - 1) and done)
   {
      sprintf(filename, "%s.%04d.%03d", "lorenz-Delta.out", index, rank);
      file.open(filename);
      pack_array(file, u->state);
      file.close();
   }

   return 0;
}

// --------------------------------------------------------------------------
// Main driver
// --------------------------------------------------------------------------

int main(int argc, char *argv[])
{
   double tstart, tstop;
   int nt, Tf_lyap, cfactor, rank;
   bool useDelta;

   // Define time domain: nt intervals
   nt = 1024;
   Tf_lyap = 2;
   tstart = 0.0;
   tstop = Tf_lyap * T_lyap;

   // MGRIT params
   cfactor = 2;
   useDelta = false;

   // Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // set up app structure
   MyBraidApp app(MPI_COMM_WORLD, rank, tstart, tstop, nt, cfactor, useDelta);

   // Initialize Braid Core Object and set some solver options
   BraidCore core(MPI_COMM_WORLD, &app);
   core.SetPrintLevel(2);
   core.SetMaxLevels(2);
   core.SetAbsTol(1.0e-10);
   core.SetCFactor(-1, cfactor);
   core.SetNRelax(-1, 0);

   // Run Simulation
   core.Drive();

   // Clean up
   MPI_Finalize();

   return (0);
}