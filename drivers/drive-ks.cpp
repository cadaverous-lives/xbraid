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
// Driver:        drive-ks.cpp
//
// Interface:     C++
//
// Requires:      C-language and C++ support, Eigen linear algebra library:   https://gitlab.com/libeigen/eigen
//                umfpack
//
// Compile with:  make drive-ks
//
// Help with:     ./drive-ks -help
//
// Sample run:    mpirun -np 2 drive-ks
//
// Description:   solve the Kuramoto-Shivasinsky equation using 4th order finite differencing and Lobatto IIIC with optional low rank Delta correction
//
//

#include <iostream>
#include <cmath>
#include <vector>
#include <string.h>
#include <fstream>

#include "braid.hpp"
#include "drive-ks-lib/ks_lib.hpp"

// --------------------------------------------------------------------------
// User-defined routines and objects
// --------------------------------------------------------------------------

// Define BraidVector, can contain anything, and be named anything
// --> Put all time-dependent information here
class BraidVector
{
public:
   VEC state;
   VEC action;
   MAT Psi;
   MAT Delta;

   // Construct a BraidVector for a given vector of doubles
   BraidVector(VEC state_, VEC prev_c_point_, MAT Psi_, MAT Delta_) : state(state_), action(prev_c_point_), Psi(Psi_), Delta(Delta_) {}
   BraidVector(int nx, int rank) : state(VEC::Zero(nx)), action(VEC::Zero(nx)), Psi(MAT::Identity(nx, rank)), Delta(MAT::Identity(nx, rank)) {}
   BraidVector() : state(VEC()), action(VEC()), Psi(MAT()), Delta(MAT()) {}

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
   int cfactor;      // Currently only supporting one CF for all levels
   int newton_iters; // only used if useTheta
   int DeltaRank;
   bool useDelta;
   bool useTheta;
   std::vector<double> thetas;

   int nx;
   KSDiscretization disc;

   // Constructor
   MyBraidApp(MPI_Comm comm_t_, int rank_, double tstart_, double tstop_, int ntime_, int cfactor_, bool useDelta_, int DeltaRank_, bool useTheta_, int newton_iters_, int max_levels, int nx, double length);

   // We will need the MPI Rank
   int rank;

   // Deconstructor
   virtual ~MyBraidApp(){};

   int IsCPoint(int i, int level);

   double getTheta(int level);

   // this step function isn't aware of the tau correction
   VEC baseStep(const VEC &u,
                const VEC &ustop,
                double dt,
                int level,
                int nlevels,
                MAT *P_tan_ptr = nullptr);

   // computes the dot product between the derivative of the step function
   // dPhi/du and the vector, v. Isn't aware of the Delta correction.
   MAT baseStepDiffDot(const MAT &v,
                       const VEC &u,
                       const VEC &ustop,
                       double dt,
                       int level,
                       int nlevels,
                       MAT *P_tan_ptr = nullptr);

   MAT LRDeltaDot(const MAT &u,
                  const MAT &Delta,
                  const MAT &Psi);

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
                        braid_Vector f_,
                        braid_Vector r_,
                        BraidStepStatus &pstatus);

   // not needed:
   virtual int Coarsen(braid_Vector fu_,
                       braid_Vector *cu_ptr,
                       BraidCoarsenRefStatus &status) { return 0; }

   virtual int Refine(braid_Vector cu_,
                      braid_Vector *fu_ptr,
                      BraidCoarsenRefStatus &status) { return 0; }
};

// Braid App Constructor
MyBraidApp::MyBraidApp(MPI_Comm comm_t_, int rank_, double tstart_, double tstop_, int ntime_, int cfactor_, bool useDelta_, int DeltaRank_, bool useTheta_, int newton_iters_, int max_levels, int nx, double length)
    : BraidApp(comm_t_, tstart_, tstop_, ntime_)
{
   rank = rank_;
   cfactor = cfactor_;
   newton_iters = newton_iters_;
   useDelta = useDelta_;
   useTheta = useTheta_;
   DeltaRank = DeltaRank_;

   // fourth order
   Stencil d1 = Stencil({1. / 12, -2. / 3, 0., 2. / 3, -1. / 12});
   Stencil d2 = Stencil({-1. / 12, 4. / 3, -5. / 2, 4. / 3, -1. / 12});
   Stencil d4 = Stencil({-1. / 6, 2., -13. / 2, 28. / 3, -13. / 2, 2., -1. / 6});
   disc = KSDiscretization(nx, length, d1, d2, d4);

   thetas.assign(max_levels, 1.);
   if (useTheta)
   {
      double total_cf;
      for (int level = 1; level < max_levels; level++)
      {
         total_cf = intpow(cfactor, level);
         // second order
         thetas[level] = (total_cf - sqrt(3 * total_cf * total_cf + 6) / 3) / total_cf; // theta2

         // fourth order
         // double cf_4 = std::pow(total_cf, 4);
         // if (cf_4 == 0.) // overflow
         // {
         //    thetas[level] = 1 - std::sqrt(10) / 5.;
         // }
         // else
         // {
         //    thetas[level] = 1. - std::sqrt(10 * cf_4 + 15) / (5 * total_cf * total_cf);
         // }
      }
   }
}

// Helper function to check if current point is a C point for this level
int MyBraidApp::IsCPoint(int i, int level)
{
   return ((i % cfactor) == 0);
}

// Helper function to compute theta for a given level
double MyBraidApp::getTheta(int level)
{
   return thetas[level];
}

VEC MyBraidApp::baseStep(const VEC &u, const VEC &ustop, double dt, int level, int nlevels, MAT *P_tan_ptr)
{
   // second order theta method
   if (level == 0 || !useTheta)
   {
      return theta2(u, ustop, disc, dt, 0., 0., 1., P_tan_ptr, newton_iters);
   }
   double theta = getTheta(level);
   double cf = intpow(cfactor, level);
   return theta2(u, ustop, disc, dt, theta, theta, 2. / 3 + 1 / (3 * cf * cf) - theta, P_tan_ptr, newton_iters);

   // fourth order theta method
   // if (level == 0 || !useTheta)
   // {
   //    return theta4(u, ustop, disc, dt, 0., 0., 1., P_tan_ptr, newton_iters, std::sqrt(disc.nx)*1e-9);
   // }
   // double cf = intpow(cfactor, level);
   // double theta = getTheta(level);
   // double cf_4 = cf*cf*cf*cf;
   // bool overflow = (cf_4 == 0.);
   // double th_C = 7./10 - theta;
   // if (!overflow)
   // {
   //    th_C += 3./(10*cf_4);
   // }
   // return theta4(u, ustop, disc, dt, theta, theta, th_C, P_tan_ptr, newton_iters);
}

MAT MyBraidApp::baseStepDiffDot(const MAT &v,
                                const VEC &u,
                                const VEC &ustop,
                                const double dt,
                                const int level,
                                const int nlevels,
                                MAT *P_tan_ptr)
{
   // use full precomputed linear tangent propagator
   if (P_tan_ptr)
   {
      return (*P_tan_ptr) * v;
   }

   // else use finite difference approximation (this is expensive)
   MAT out = v;
   const double eps = 1e-8;

   for (Eigen::Index i = 0; i < v.cols(); i++)
   {
      // TODO: can I get a better initial guess than ustop + eps*col?
      out.col(i) = (baseStep(u + eps * v.col(i), ustop + eps * v.col(i), dt, level, nlevels) - ustop) / eps;
   }
   return out;
}

MAT MyBraidApp::LRDeltaDot(const MAT &u,
                           const MAT &Delta,
                           const MAT &Psi)
{
   return Delta * (Psi.transpose() * u);
}

//
int MyBraidApp::Step(braid_Vector u_,
                     braid_Vector ustop_,
                     braid_Vector fstop_,
                     BraidStepStatus &pstatus)
{

   BraidVector *u = (BraidVector *)u_;
   BraidVector *f = (BraidVector *)fstop_;
   BraidVector *ustop = (BraidVector *)ustop_;

   double tstart; // current time
   double tstop;  // evolve to this time
   int level, nlevels, T_index, calling_fnc;

   pstatus.GetTstartTstop(&tstart, &tstop);
   pstatus.GetLevel(&level);
   pstatus.GetNLevels(&nlevels);
   pstatus.GetTIndex(&T_index); // this is the index of tstart
   pstatus.GetCallingFunction(&calling_fnc);

   double dt = tstop - tstart;

   // no refinement
   pstatus.SetRFactor(1);

   bool computeDeltas = (calling_fnc == braid_ASCaller_FRestrict); // only compute Deltas when in FRestrict
   bool normalize;
   // only want to normalize at C-points, or on the coarsest grid
   normalize = (IsCPoint(T_index + 1, level) || level == nlevels - 1);
   normalize = (normalize || calling_fnc == braid_ASCaller_FInterp); // or when in finterp

   // std::cout << "Stp called @ " << T_index << " on level " << level << '\n';
   // std::cout << "f is null: " << f_is_null << '\n';

   VEC utmp(u->state);
   MAT Psitmp(u->Psi);
   utmp = baseStep(u->state, ustop->state, dt, level, nlevels, &Psitmp);

   if (!useDelta) // default behavior, no Delta correction
   {
      if (f)
      {
         utmp += f->state;
         Psitmp += f->Psi; // here the tau correction for Psi is nonzero, since we are not using Delta correction
      }
      // orthonormalize lyapunov vectors at C-points
      if (normalize)
      {
         GramSchmidt(Psitmp);
      }
      u->state = utmp;
      u->Psi = Psitmp;
      return 0;
   }
   // else:
   if (f)
   {
      // tau = state - action
      utmp += LRDeltaDot(u->state, f->Delta, f->Psi) + f->state - f->action;
      // when using low rank Delta correction, no tau correction is needed for Psi
      Psitmp += LRDeltaDot(u->Psi, f->Delta, f->Psi);
   }

   // store state and Psi at previous C-point
   if (computeDeltas)
   {
      if (IsCPoint(T_index, level))
      {
         // Need to store the value at the previous c-point for tau correction later
         u->action = u->state;
         u->Delta = u->Psi;
      }
   }

   // normalize Psi at c-points:
   if (normalize)
   {
      GramSchmidt(Psitmp);
   }

   u->state = utmp;
   u->Psi = Psitmp;

   return 0;
}

int MyBraidApp::Residual(braid_Vector u_,
                         braid_Vector f_,
                         braid_Vector r_,
                         BraidStepStatus &pstatus)
{
   BraidVector *u = (BraidVector *)u_;
   BraidVector *f = (BraidVector *)f_;
   BraidVector *r = (BraidVector *)r_;

   double tstart; // current time
   double tstop;  // evolve to this time
   int level, nlevels, T_index, calling_fnc;

   pstatus.GetTstartTstop(&tstart, &tstop);
   pstatus.GetLevel(&level);
   pstatus.GetNLevels(&nlevels);
   pstatus.GetTIndex(&T_index);
   pstatus.GetCallingFunction(&calling_fnc);

   double dt = tstop - tstart;

   VEC utmp(r->state);
   MAT Psitmp(r->Psi);
   utmp = baseStep(r->state, u->state, dt, level, nlevels, &Psitmp);

   if (!useDelta)
   {
      if (f)
      { // do tau correction
         utmp += f->state;
         Psitmp += f->Psi;
      }

      r->state = u->state - utmp;
      r->Psi = u->Psi - Psitmp;

      return 0;
   }
   // else:

   if (calling_fnc == braid_ASCaller_Residual)
   { // this is called on the coarse grid right after restriction
      r->action = -LRDeltaDot(r->state, Psitmp, r->Psi);
      r->Delta = -Psitmp;
      r->state = u->state - utmp;
      r->Psi.setZero();
      return 0;
   }

   // else this is called on the fine grid right after F-relax
   if (f)
   { // do delta correction and tau correction
      Psitmp += LRDeltaDot(r->Psi, f->Delta, f->Psi);
      utmp += LRDeltaDot(r->state, f->Delta, f->Psi) + (f->state - f->action);
   }

   r->state = u->state - utmp;                           // u_i - Phi^m(u_{i-m})
   r->action = -LRDeltaDot(r->action, Psitmp, r->Delta); // -([D Phi^m] \Psi_{i-m}) \Psi_{i-m}^T u_{i-m}
   r->Psi = -r->Delta;                                   // -\Psi_{i-m}
   r->Delta = -Psitmp;                                   // -[D \Phi^m] \Psi_{i-m}
   return 0;
}

int MyBraidApp::Init(double t,
                     braid_Vector *u_ptr)
{
   // this should take care of most of the initialization
   BraidVector *u = new BraidVector(disc.nx, DeltaRank);
   setFourierMatrix(u->Psi, disc.nx, disc.len);

   if (t == tstart)
   {
      // set initial condition
      // u->state = FourierMode(2, disc.nx, disc.len);
      u->state = smoothed_noise(disc.nx, disc.nx/8);
      u->action = u->state;
   }

   *u_ptr = (braid_Vector)u;
   return 0;
}

int MyBraidApp::Clone(braid_Vector u_,
                      braid_Vector *v_ptr)
{
   // std::cout << "Clone called" << '\n';

   BraidVector *u = (BraidVector *)u_;
   BraidVector *v = new BraidVector(u->state, u->action, u->Psi, u->Delta);
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
   // std::cout << "Sum called" << '\n';
   BraidVector *x = (BraidVector *)x_;
   BraidVector *y = (BraidVector *)y_;
   (y->state) = alpha * (x->state) + beta * (y->state);
   (y->Psi) = alpha * (x->Psi) + beta * (y->Psi);
   if (useDelta)
   {
      (y->action) = alpha * (x->action) + beta * (y->action);
      (y->Delta) = alpha * (x->Delta) + beta * (y->Delta);
   }
   return 0;
}

int MyBraidApp::SpatialNorm(braid_Vector u_,
                            double *norm_ptr)
{
   // std::cout << "Norm called" << '\n';
   BraidVector *u = (BraidVector *)u_;
   *norm_ptr = u->state.norm()/std::sqrt(disc.nx); // normalized like l2 norm
   return 0;
}

int MyBraidApp::BufSize(int *size_ptr,
                        BraidBufferStatus &status)
{
   *size_ptr = (2 * disc.nx + 2 * disc.nx * DeltaRank) * sizeof(double);
   return 0;
}

int MyBraidApp::BufPack(braid_Vector u_,
                        void *buffer,
                        BraidBufferStatus &status)
{
   BraidVector *u = (BraidVector *)u_;
   double *dbuffer = (double *)buffer;
   // std::cout << "buffpack called" << '\n';
   size_t bf_size = 0;

   bf_pack_help(dbuffer, u->state, disc.nx, bf_size);
   // std::cout << "state successful\n";
   bf_pack_help(dbuffer, u->Psi, disc.nx * DeltaRank, bf_size);
   // std::cout << "lyapunov successful\n";

   if (useDelta)
   {
      bf_pack_help(dbuffer, u->action, disc.nx, bf_size);
      bf_pack_help(dbuffer, u->Delta, disc.nx * DeltaRank, bf_size);
   }
   status.SetSize(bf_size * sizeof(double));
   return 0;
}

int MyBraidApp::BufUnpack(void *buffer,
                          braid_Vector *u_ptr,
                          BraidBufferStatus &status)
{
   double *dbuffer = (double *)buffer;
   // std::cout << "buffunpack called" << '\n';

   BraidVector *u = new BraidVector(disc.nx, DeltaRank);

   size_t bf_size = 0;

   bf_unpack_help(dbuffer, u->state, disc.nx, bf_size);
   bf_unpack_help(dbuffer, u->Psi, disc.nx * DeltaRank, bf_size);

   if (useDelta)
   {
      bf_unpack_help(dbuffer, u->action, disc.nx, bf_size);
      bf_unpack_help(dbuffer, u->Delta, disc.nx * DeltaRank, bf_size);
   }
   *u_ptr = (braid_Vector)u;

   return 0;
}

int MyBraidApp::Access(braid_Vector u_,
                       BraidAccessStatus &astatus)
{
   // std::cout << "Access called" << '\n';
   char filename[255];
   char lv_fname[255];
   std::ofstream file;
   BraidVector *u = (BraidVector *)u_;

   // Extract information from astatus
   int done, level, iter, index;
   double t;
   astatus.GetTILD(&t, &iter, &level, &done);
   astatus.GetTIndex(&index);
   // int nt = MyBraidApp::ntime;

   // Print information to file
   if (done && level == 0 && IsCPoint(index, level))
   {
      sprintf(filename, "%s.%04d", "drive-ks.out", index);
      file.open(filename);
      pack_array(file, u->state);
      file.close();

      sprintf(lv_fname, "%s.%04d", "drive-ks-lv.out", index);
      file.open(lv_fname);
      pack_darray(file, u->Psi);
      file.close();
   }

   return 0;
}

// --------------------------------------------------------------------------
// Main driver
// --------------------------------------------------------------------------

int del_extra(int nt, int cfactor, std::string fname)
{
   std::ifstream inf;
   char ifname[255];
   // loop over all possible files with that name
   for (int i = 0; i <= nt; i += cfactor)
   {
      // check if they exist
      sprintf(ifname, "%s.%s.%04d", fname.c_str(), "out", i);
      inf.open(ifname);
      if (inf)
      {
         // delete if they do
         inf.close();
         remove(ifname);
      }
   }
   return 0;
}

int collate_files(int nt, int cfactor, std::string fname)
{
   // collate files:
   std::ifstream inf;
   std::ofstream of;
   char ifname[255];
   char ofname[255];

   // state vectors:
   sprintf(ofname, "%s.out", fname.c_str());
   of.open(ofname);
   for (int i = 0; i <= nt; i += cfactor)
   {
      sprintf(ifname, "%s.%s.%04d", fname.c_str(), "out", i);
      inf.open(ifname);
      // if we can't open one file, delete all extra files
      if (!inf)
      {
         std::cout << "!!!error collating files- deleting extra!!!\n";
         del_extra(nt, cfactor, fname);
         return 1;
      }
      of << inf.rdbuf();
      inf.close();
      // delete the extra files
      remove(ifname);
   }
   of.close();
   return 0;
}

int main(int argc, char *argv[])
{
   double tstart, tstop;
   int rank;

   int nt = 256;
   double Tf_lyap = 4;
   int max_levels = 2;
   int nrelax = 0;
   int nrelax0 = 0;
   double tol = 1e-6;
   int cfactor = 4;
   int max_iter = 25;
   int newton_iters = 10;
   bool useFMG = false;
   bool useDelta = false;
   bool useTheta = false;
   bool wrapperTests = false;
   bool output = false;
   int DeltaRank = 1;

   // KS parameters
   double len = 64;
   int nx = 64;

   int arg_index;

   // Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // Parse command line
   arg_index = 1;
   while (arg_index < argc)
   {
      if (strcmp(argv[arg_index], "-help") == 0)
      {
         if (rank == 0)
         {
            printf("\n");
            printf("  -nt         : set num time points (default %d)\n", nt);
            printf("  -nx         : set num space points (default %d)\n", nx);
            printf("  -len        : set length of spatial domain (default %f)\n", len);
            printf("  -tf         : set end time, in Lyapunov time (default %lf)\n", Tf_lyap);
            printf("  -ml         : set max levels\n");
            printf("  -nu         : set num F-C relaxations\n");
            printf("  -nu0        : set num F-C relaxations on level 0\n");
            printf("  -tol        : set stopping tolerance\n");
            printf("  -cf         : set coarsening factor\n");
            printf("  -mi         : set max iterations\n");
            printf("  -niters     : set number of newton iters for theta method\n");
            printf("  -fmg        : use FMG cycling\n");
            printf("  -Delta      : use delta correction\n");
            printf("  -rank       : set rank of delta correction (Default: 3)\n");
            printf("  -theta      : use first order theta method\n");
            printf("  -out        : write output to file (for visualization)\n");
            printf("  -test       : run wrapper tests\n");
            printf("\n");
         }
         exit(0);
      }
      else if (strcmp(argv[arg_index], "-nt") == 0)
      {
         arg_index++;
         nt = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-nx") == 0)
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-len") == 0)
      {
         arg_index++;
         len = atof(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-tf") == 0)
      {
         arg_index++;
         Tf_lyap = atof(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-ml") == 0)
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-nu") == 0)
      {
         arg_index++;
         nrelax = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-nu0") == 0)
      {
         arg_index++;
         nrelax0 = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-tol") == 0)
      {
         arg_index++;
         tol = atof(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-cf") == 0)
      {
         arg_index++;
         cfactor = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-mi") == 0)
      {
         arg_index++;
         max_iter = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-niters") == 0)
      {
         arg_index++;
         newton_iters = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-fmg") == 0)
      {
         arg_index++;
         useFMG = true;
      }
      else if (strcmp(argv[arg_index], "-Delta") == 0)
      {
         arg_index++;
         useDelta = true;
      }
      else if (strcmp(argv[arg_index], "-rank") == 0)
      {
         arg_index++;
         DeltaRank = atoi(argv[arg_index++]);
      }
      else if (strcmp(argv[arg_index], "-theta") == 0)
      {
         arg_index++;
         useTheta = true;
      }
      else if (strcmp(argv[arg_index], "-out") == 0)
      {
         arg_index++;
         output = true;
      }
      else if (strcmp(argv[arg_index], "-test") == 0)
      {
         arg_index++;
         wrapperTests = true;
      }
      else
      {
         arg_index++;
         /*break;*/
      }
   }

   tstart = 0.0;
   tstop = Tf_lyap * T_lyap;

   if (useDelta && rank == 0)
   {
      std::cout << "Using Delta correction\n";
   }
   if (useTheta && rank == 0)
   {
      std::cout << "Using theta method\n";
   }

   // set up app structure
   MyBraidApp app(MPI_COMM_WORLD, rank, tstart, tstop, nt, cfactor, useDelta, DeltaRank, useTheta, newton_iters, max_levels, nx, len);

   // wrapper tests
   if (wrapperTests)
   {
      if (rank != 0)
      {
         // Clean up
         MPI_Finalize();
         return 0;
      }
      BraidUtil Util = BraidUtil();
      // FILE *ftest = fopen("KSwrapperTests.txt", "w");
      Util.TestInitAccess(&app, MPI_COMM_WORLD, stdout, 0.);
      Util.TestInitAccess(&app, MPI_COMM_WORLD, stdout, 1.);
      Util.TestClone(&app, MPI_COMM_WORLD, stdout, 0.);
      Util.TestSpatialNorm(&app, MPI_COMM_WORLD, stdout, 0.);
      Util.TestBuf(&app, MPI_COMM_WORLD, stdout, 0.);
      Util.TestResidual(&app, MPI_COMM_WORLD, stdout, 1., 0.01);
      
      // Clean up
      MPI_Finalize();
      return 0;
   }

   // Initialize Braid Core Object and set some solver options
   BraidCore core(MPI_COMM_WORLD, &app);
   core.SetResidual();
   if (useFMG)
   {
      core.SetFMG();
      core.SetNFMG(2);
   }
   core.SetPrintLevel(2);
   core.SetMaxLevels(max_levels);
   core.SetMaxIter(max_iter);
   core.SetAbsTol(tol);
   core.SetCFactor(-1, cfactor);
   core.SetNRelax(-1, nrelax);
   core.SetNRelax(0, nrelax0);
   core.SetSkip(1);
   core.SetStorage(0);
   core.SetTemporalNorm(2);

   if (output)
   {
      core.SetAccessLevel(1);
   }
   else
   {
      core.SetAccessLevel(0);
   }

   // Run Simulation
   core.Drive();

   if (rank == 0 && output)
   {
      collate_files(nt, cfactor, "drive-ks");
      collate_files(nt, cfactor, "drive-ks-lv");
   }

   // Clean up
   MPI_Finalize();

   return (0);
}