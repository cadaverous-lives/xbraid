// Copyright (c) 2013, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. Written by
// Jacob Schroder schroder2@llnl.gov, Rob Falgout falgout2@llnl.gov,
// Tzanio Kolev kolev1@llnl.gov, Ulrike Yang yang11@llnl.gov,
// Veselin Dobrev dobrev1@llnl.gov, et al.
// LLNL-CODE-660355. All rights reserved.
//
// This file is part of XBraid. Email schroder2@llnl.gov on how to download.
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this program; if not, write to the Free Software Foundation, Inc., 59
// Temple Place, Suite 330, Boston, MA 02111-1307 USA
//

#ifndef braid_hpp_HEADER
#define braid_hpp_HEADER

#include "_braid.h"
#include "braid.h"
#include "braid_test.h"

// Wrapper for BRAID's App object. Users should inherit this class and implement
// the purely virtual functions (see braid.h for descriptions).
class BraidApp
{
public:
   MPI_Comm comm_t;
   double   tstart;
   double   tstop;
   int      ntime;

   BraidApp(MPI_Comm _comm_t, double _tstart = 0.0, double _tstop = 1.0, int _ntime = 100)
      : comm_t(_comm_t), tstart(_tstart), tstop(_tstop), ntime(_ntime) { }

   ~BraidApp() { }

   virtual int Phi(braid_App       _app,
                   braid_Vector    _u,
                   braid_PhiStatus _pstatus) = 0;

   virtual int Clone(braid_App     _app,
                     braid_Vector  _u,
                     braid_Vector *v_ptr) = 0;

   virtual int Init(braid_App    _app,
                    double       t,
                    braid_Vector *u_ptr) = 0;

   virtual int Free(braid_App    _app,
                    braid_Vector _u) = 0;

   virtual int Sum(braid_App    _app,
                   double       alpha,
                   braid_Vector _x,
                   double       beta,
                   braid_Vector _y) = 0;

   virtual int SpatialNorm(braid_App     _app,
                           braid_Vector  _u,
                           double       *norm_ptr) = 0;

   virtual int Access(braid_App           _app,
                      braid_Vector        _u,
                      braid_AccessStatus  _astatus) = 0;

   virtual int BufSize(braid_App  _app,
                       int       *size_ptr) = 0;

   virtual int BufPack(braid_App     _app,
                       braid_Vector  _u,
                       void         *buffer,
                       int          *size_ptr) = 0;

   virtual int BufUnpack(braid_App     _app,
                         void         *buffer,
                         braid_Vector *u_ptr) = 0;
};

// Static functions passed to Braid, with braid_App == BraidApp*
static int _BraidAppPhi(braid_App       _app,
                        braid_Vector    _u,
                        braid_PhiStatus _pstatus)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> Phi(_app, _u, _pstatus);
}

static int _BraidAppClone(braid_App     _app,
                          braid_Vector  _u,
                          braid_Vector *v_ptr)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> Clone(_app, _u, v_ptr);
}

static int _BraidAppInit(braid_App    _app,
                         double       t,
                         braid_Vector *u_ptr)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> Init(_app, t, u_ptr);
}

static int _BraidAppFree(braid_App    _app,
                         braid_Vector _u)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> Free(_app, _u);
}

static int _BraidAppSum(braid_App    _app,
                        double       alpha,
                        braid_Vector _x,
                        double       beta,
                        braid_Vector _y)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> Sum(_app, alpha, _x, beta, _y);
}

static int _BraidAppSpatialNorm(braid_App     _app,
                                braid_Vector  _u,
                                double       *norm_ptr)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> SpatialNorm(_app, _u, norm_ptr);
}

static int _BraidAppAccess(braid_App           _app,
                           braid_Vector        _u,
                           braid_AccessStatus  _astatus)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> Access(_app, _u, _astatus);
}

static int _BraidAppBufSize(braid_App  _app,
                            int       *size_ptr)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> BufSize(_app, size_ptr);
}

static int _BraidAppBufPack(braid_App     _app,
                            braid_Vector  _u,
                            void         *buffer,
                            int          *size_ptr)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> BufPack(_app, _u, buffer, size_ptr);
}

static int _BraidAppBufUnpack(braid_App     _app,
                              void         *buffer,
                              braid_Vector *u_ptr)
{
   BraidApp *app = (BraidApp*)_app;
   return app -> BufUnpack(_app, buffer, u_ptr);
}

// Wrapper for BRAID's core object
class BraidCore
{
private:
   braid_Core core;

public:
   BraidCore(MPI_Comm comm_world, BraidApp *app)
   {
      braid_Init(comm_world,
                 app->comm_t, app->tstart, app->tstop, app->ntime, (braid_App)app,
                 _BraidAppPhi, _BraidAppInit, _BraidAppClone, _BraidAppFree,
                 _BraidAppSum, _BraidAppSpatialNorm, _BraidAppAccess,
                 _BraidAppBufSize, _BraidAppBufPack, _BraidAppBufUnpack, &core);
   }

   void SetMaxLevels(int max_levels) { braid_SetMaxLevels(core, max_levels); }

   void SetMaxCoarse(int max_coarse) { braid_SetMaxCoarse(core, max_coarse); }

   void SetNRelax(int level, int nrelax)
   { braid_SetNRelax(core, level, nrelax); }

   void SetAbsTol(double tol) { braid_SetAbsTol(core, tol); }

   void SetRelTol(double tol) { braid_SetRelTol(core, tol); }

   void SetTemporalNorm(int tnorm) { braid_SetTemporalNorm(core, tnorm); }

   void SetCFactor(int level, int cfactor)
   { braid_SetCFactor(core, level, cfactor); }

   /** Use cfactor0 on all levels until there are < cfactor0 points
       on each processor. */
   void SetAggCFactor(int cfactor0)
   {
      BraidApp *app = (BraidApp *) core->app;
      int nt = app->ntime, pt;
      MPI_Comm_size(app->comm_t, &pt);
      if (cfactor0 > -1)
      {
         int level = (int) (log10((nt + 1) / pt) / log10(cfactor0));
         for (int i = 0; i < level; i++)
            braid_SetCFactor(core, i, cfactor0);
      }
   }

   void SetMaxIter(int max_iter) { braid_SetMaxIter(core, max_iter); }

   void SetPrintLevel(int print_level) { braid_SetPrintLevel(core, print_level); }

   void SetPrintFile(const char *printfile_name) { braid_SetPrintFile(core, printfile_name); }

   void SetAccessLevel(int access_level) { braid_SetAccessLevel(core, access_level); }

   void SetFMG() { braid_SetFMG(core); }

   void SetNFMGVcyc(int nfmg_Vcyc) { braid_SetNFMGVcyc(core, nfmg_Vcyc); }

   void GetNumIter(int *niter_ptr) { braid_GetNumIter(core, niter_ptr); }

   void GetRNorm(double *rnorm_ptr) { braid_GetRNorm(core, rnorm_ptr); }

   void Drive() { braid_Drive(core); }

   ~BraidCore() { braid_Destroy(core); }
};


// Wrapper for BRAID's AccessStatus object
class BraidAccessStatus
{
   private:
      braid_AccessStatus astatus;

   public:
      BraidAccessStatus(braid_AccessStatus _astatus)
      {
         astatus = _astatus;
      }

      void GetTILD(braid_Real *t_ptr, braid_Int *iter_ptr, braid_Int *level_ptr, braid_Int *done_ptr) { braid_AccessStatusGetTILD(astatus, t_ptr, iter_ptr, level_ptr, done_ptr); }
      void GetT(braid_Real *t_ptr)            { braid_AccessStatusGetT(astatus, t_ptr); }
      void GetDone(braid_Int *done_ptr)       { braid_AccessStatusGetDone(astatus, done_ptr); }
      void GetLevel(braid_Int *level_ptr)     { braid_AccessStatusGetLevel(astatus, level_ptr); }
      void GetIter(braid_Int *iter_ptr)       { braid_AccessStatusGetIter(astatus, iter_ptr); }
      void GetResidual(braid_Real *rnorm_ptr) { braid_AccessStatusGetResidual(astatus, rnorm_ptr); }

      // The braid_AccessStatus structure is deallocated inside of Braid
      // This class is just to make code consistently look object oriented
      ~BraidAccessStatus() { }
};

// Wrapper for BRAID's PhiStatus object
class BraidPhiStatus
{
   private:
      braid_PhiStatus pstatus;

   public:
      BraidPhiStatus(braid_PhiStatus _pstatus)
      {
         pstatus = _pstatus;
      }

      void GetTstartTstop(braid_Real *tstart_ptr, braid_Real *tstop_ptr)     { braid_PhiStatusGetTstartTstop(pstatus, tstart_ptr, tstop_ptr); }
      void GetTstart(braid_Real *tstart_ptr)     { braid_PhiStatusGetTstart(pstatus, tstart_ptr); }
      void GetTstop(braid_Real *tstop_ptr)       { braid_PhiStatusGetTstop(pstatus, tstop_ptr); }
      void GetAccuracy(braid_Real *accuracy_ptr)  { braid_PhiStatusGetAccuracy(pstatus, accuracy_ptr); }
      void SetRFactor(braid_Int rfactor)         { braid_PhiStatusSetRFactor(pstatus, rfactor); }

      // The braid_PhiStatus structure is deallocated inside of Braid
      // This class is just to make code consistently look object oriented
      ~BraidPhiStatus() { }
};

// Wrapper for BRAID's CoarsenRefStatus object
class BraidCoarsenRefStatus
{
   private:
      braid_CoarsenRefStatus cstatus;

   public:
      BraidCoarsenRefStatus(braid_CoarsenRefStatus  _cstatus)
      {
         cstatus = _cstatus;
      }

      void GetTpriorTstop(braid_Real *tstart_ptr, braid_Real *f_tprior_ptr, braid_Real *f_tstop_ptr, braid_Real *c_tprior_ptr, braid_Real *c_tstop_ptr)     { braid_CoarsenRefStatusGetTpriorTstop(cstatus, tstart_ptr, f_tprior_ptr, f_tstop_ptr, c_tprior_ptr, c_tstop_ptr); }
      void GetTstart(braid_Real *tstart_ptr)     { braid_CoarsenRefStatusGetTstart(cstatus, tstart_ptr); }
      void GetFTstop(braid_Real *f_tstop_ptr)    { braid_CoarsenRefStatusGetFTstop(cstatus, f_tstop_ptr); }
      void GetFTprior(braid_Real *f_tprior_ptr)  { braid_CoarsenRefStatusGetFTprior(cstatus, f_tprior_ptr); }
      void GetCTstop(braid_Real *c_tstop_ptr)    { braid_CoarsenRefStatusGetCTstop(cstatus, c_tstop_ptr); }
      void GetCTprior(braid_Real *c_tprior_ptr)  { braid_CoarsenRefStatusGetCTprior(cstatus, c_tprior_ptr); }

      // The braid_CoarsenRefStatus structure is deallocated inside of Braid
      // This class is just to make code consistently look object oriented
      ~BraidCoarsenRefStatus() { }
};

// Wrapper for BRAID utilities that help the user,
// includes all the braid_Test* routines for testing the
// user-written wrappers.
class BraidUtil
{
private:

public:

   // Empty constructor
   BraidUtil( ){ }

   // Split comm_world into comm_x and comm_t, the spatial
   // and temporal communicators
   void SplitCommworld(const MPI_Comm  *comm_world,
                             braid_Int  px,
                             MPI_Comm  *comm_x,
                             MPI_Comm  *comm_t)
   { braid_SplitCommworld(comm_world, px, comm_x, comm_t); }

   // Test Function for Init and Access function
   void TestInitAccess( BraidApp              *app,
                        MPI_Comm               comm_x,
                        FILE                  *fp,
                        double                 t,
                        braid_PtFcnInit        init,
                        braid_PtFcnAccess      access,
                        braid_PtFcnFree        free)
   { braid_TestInitAccess((braid_App) app, comm_x, fp, t, init, access, free); }

   // Test Function for Clone
   void TestClone( BraidApp              *app,
                   MPI_Comm               comm_x,
                   FILE                  *fp,
                   double                 t,
                   braid_PtFcnInit        init,
                   braid_PtFcnAccess      access,
                   braid_PtFcnFree        free,
                   braid_PtFcnClone       clone)
   { braid_TestClone((braid_App) app, comm_x, fp, t, init, access, free, clone); }

   // Test Function for Sum
   void TestSum( BraidApp              *app,
                 MPI_Comm               comm_x,
                 FILE                  *fp,
                 double                 t,
                 braid_PtFcnInit        init,
                 braid_PtFcnAccess      access,
                 braid_PtFcnFree        free,
                 braid_PtFcnClone       clone,
                 braid_PtFcnSum         sum)
   { braid_TestSum((braid_App) app, comm_x, fp, t, init, access, free, clone, sum); }

   // Test Function for SpatialNorm
   int TestSpatialNorm( BraidApp              *app,
                        MPI_Comm               comm_x,
                        FILE                  *fp,
                        double                 t,
                        braid_PtFcnInit        init,
                        braid_PtFcnFree        free,
                        braid_PtFcnClone       clone,
                        braid_PtFcnSum         sum,
                        braid_PtFcnSpatialNorm spatialnorm)
   { return braid_TestSpatialNorm((braid_App) app, comm_x, fp, t, init, free, clone, sum, spatialnorm); }

   // Test Functions BufSize, BufPack, BufUnpack
   int TestBuf( BraidApp               *app,
                 MPI_Comm               comm_x,
                 FILE                  *fp,
                 double                 t,
                 braid_PtFcnInit        init,
                 braid_PtFcnFree        free,
                 braid_PtFcnSum         sum,
                 braid_PtFcnSpatialNorm spatialnorm,
                 braid_PtFcnBufSize     bufsize,
                 braid_PtFcnBufPack     bufpack,
                 braid_PtFcnBufUnpack   bufunpack)
   { return braid_TestBuf((braid_App) app, comm_x, fp, t, init, free, sum, spatialnorm, bufsize, bufpack, bufunpack); }

   // Test Functions Coarsen and Refine
   int TestCoarsenRefine(BraidApp                 *app,
                          MPI_Comm                 comm_x,
                          FILE                    *fp,
                          double                   t,
                          double                   fdt,
                          double                   cdt,
                          braid_PtFcnInit          init,
                          braid_PtFcnAccess        access,
                          braid_PtFcnFree          free,
                          braid_PtFcnClone         clone,
                          braid_PtFcnSum           sum,
                          braid_PtFcnSpatialNorm   spatialnorm,
                          braid_PtFcnCoarsen       coarsen,
                          braid_PtFcnRefine        refine)
   { return braid_TestCoarsenRefine( (braid_App) app, comm_x, fp, t, fdt, cdt, init,
                            access, free, clone, sum, spatialnorm, coarsen, refine); }

   int TestAll(BraidApp                 *app,
                MPI_Comm                 comm_x,
                FILE                    *fp,
                double                   t,
                double                   fdt,
                double                   cdt,
                braid_PtFcnInit          init,
                braid_PtFcnFree          free,
                braid_PtFcnClone         clone,
                braid_PtFcnSum           sum,
                braid_PtFcnSpatialNorm   spatialnorm,
                braid_PtFcnBufSize       bufsize,
                braid_PtFcnBufPack       bufpack,
                braid_PtFcnBufUnpack     bufunpack,
                braid_PtFcnCoarsen       coarsen,
                braid_PtFcnRefine        refine)
   { return braid_TestAll( (braid_App) app, comm_x, fp, t, fdt, cdt,
                   init, free, clone, sum, spatialnorm, bufsize, bufpack,
                   bufunpack, coarsen, refine); }

   ~BraidUtil() { }

};

#endif