#include <boost/numpy.hpp>
#include <cmath>
#include "mjc/core/slcp.h"
#include "macros.h"
#include <iostream>
#include "mjc/core/mj_render.h"
#include <boost/python/slice.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

template<typename T>
bn::ndarray toNdarray1(const T* data, long dim0) {
  long dims[1] = {dim0};
  bn::ndarray out = bn::empty(1, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray2(const T* data, long dim0, long dim1) {
  long dims[2] = {dim0,dim1};
  bn::ndarray out = bn::empty(2, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray3(const T* data, long dim0, long dim1, long dim2) {
  long dims[3] = {dim0,dim1,dim2};
  bn::ndarray out = bn::empty(3, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*dim2*sizeof(T));
  return out;
}

enum ContactType {
  ContactType_SPRING=0,
  ContactType_SLCP=1,
  ContactType_SPRINGTAU=2,
  ContactType_CONVEX=3
};


#define  NV (m_model->ndof)
#define  NQ (m_model->nqpos)
#define  NBODY (m_model->nbody)
#define  NL (m_model->nlmax)
#define  NC (m_model->ncmax)

const int Integrator_FWD=1;

class PyMJCWorld {


public:

    PyMJCWorld(const std::string& binfile);
    bn::ndarray Step(const bn::ndarray& x, const bn::ndarray& u);
    bn::ndarray StepMulti(const bn::ndarray& x, const bn::ndarray& u);
    bp::tuple StepMulti2(const bn::ndarray& x, const bn::ndarray& u, const bn::ndarray& done);
    bp::tuple StepJacobian(const bn::ndarray& x, const bn::ndarray& u);
    void Plot(const bn::ndarray& x);    
    void SetActuatedDims(const bp::object& dims);
    void SetContactType(ContactType contactType) {
      m_contactType = contactType;
    }
    void SetTimestep(mjtNum dt) {
      m_model->timestep = dt;
    }
    bn::ndarray ComputeContacts(const bn::ndarray& x);
    bp::dict GetModel();
    void SetModel(bp::dict);
    bp::dict GetData(const bn::ndarray& x);

    bn::ndarray GetImage(const bn::ndarray& x);

    ~PyMJCWorld();
private:
    // PyMJCWorld(const PyMJCWorld&) {}

    void _PlotInit();
    mjtByte _StepLowLevel(const mjtByte flgDiff, const mjtNum* tu,
          mjtNum* contact_scratch, mjtNum *f_ctr,
          mjtNum *plam, mjtNum *pdist, mjtNum *pdebug, mjtNum *s, 
          mjtNum *jac, mjtNum *f_j);

    void _PreProcess();
    mjtNum _ContactImpulse(mjtNum* contact_scratch, mjtNum *f_ctr, const mjtNum *qvel_next_user);

    void _SetState(const mjtNum* xdata) {mju_copy(m_data->qpos, (xdata), NQ); mju_copy(m_data->qvel, (xdata)+NQ, NV); }
    void _SetControl(const mjtNum* udata) {for (int i=0; i < m_actuatedDims.size(); ++i) m_u[m_actuatedDims[i]] = (udata)[i];}
    void _ComputeCOM(mjtNum* out);


    mjModel* m_model;
    mjData* m_data;
    mjRender* m_mjr;
    std::vector<mjtNum> m_u;
    std::vector<int> m_actuatedDims;
    ContactType m_contactType;
    int m_integrator;

    mjtByte m_flagPrehoc;

    mjtNum m_spring_kspring, m_spring_kdamp;
    mjtByte m_spring_scale;

    mjtNum m_st_tau, m_st_beta; // recovery time, softness

    mjtNum m_slcp_tau, m_slcp_beta, m_slcp_tau2;   // ? ? ?
    int m_slcp_iterCloud;
    int m_slcp_maxIter;
    mjtNum* m_slcp_betaVec; // XXX i'm not sure what this is.


    mjtByte m_flagPlanar;
    double *pDBG, *pXIN, *pUIN, *pXOUT, *pLAM, *pDIST, *pS, *pJAC, *pF_J;
    double *pPUSH_BODY, *pPUSH_POINT, *pPUSH_FORCE;
    int NS, NDS;
    int m_SZ_F, m_SZ_DIST, m_SZ_DISTX;
    // int FLG_CONTACT, FLG_FWD;
    double* FEAT;
    mjtNum* cscratch[1];

};

PyMJCWorld::PyMJCWorld(const std::string& binfile) :
  m_model(mj_loadModel(binfile.c_str())),
  m_data(mj_makeData(m_model)),
  m_mjr(NULL),
  m_u(NV, 0),
  m_actuatedDims(),
  
  m_contactType(ContactType_SPRING),
  m_integrator(Integrator_FWD),
  m_flagPrehoc(true),

  m_spring_kspring(0),
  m_spring_kdamp(0),
  m_spring_scale(false),

  m_st_tau(0.04), // depth of penetration
  m_st_beta(0.1),

  m_slcp_tau(0.04), // depth of penetration
  m_slcp_beta(0.001),
  m_slcp_tau2(0.04), 
  m_slcp_iterCloud(2), // ??
  m_slcp_maxIter(10),
  m_slcp_betaVec(new mjtNum[(m_flagPlanar?2:4)*NC+NL]),

  m_flagPlanar(((m_model->jnt_type[0]==mjSLIDE)&&(m_model->jnt_type[1]==mjSLIDE)&&(m_model->jnt_type[2]==mjHINGE))),

  pDBG(NULL), 
  pXIN(NULL), 
  pUIN(NULL), 
  pXOUT(NULL), 
  pLAM(NULL), 
  pDIST(NULL), 
  pS(NULL), 
  pJAC(NULL), 
  pF_J(NULL),
  pPUSH_BODY(NULL), 
  pPUSH_POINT(NULL), 
  pPUSH_FORCE(NULL),

  NS(0),
  NDS(0),

  m_SZ_F(mjConDim*NC+NL),
  m_SZ_DIST(NC+NL),
  m_SZ_DISTX(4*NC+NL),

  FEAT(NULL)
{




    _PreProcess();

    printf("Damping[0]: %f\n", m_model->jnt_vel_damping[0]);
    printf("Gravity: %f %f %f\n", m_model->gravity[0], m_model->gravity[1], m_model->gravity[2]);
    printf("Viscosity: %f\n", m_model->viscosity);
    printf("Timestep: %f\n", m_model->timestep);
    


}

PyMJCWorld::~PyMJCWorld() {
  if (m_mjr) {
    m_mjr->Close();
    delete m_mjr;
  }
  mj_delete(m_model, m_data);
  delete[] m_slcp_betaVec;
}

#define MJTNUM_DTYPE bn::dtype::get_builtin<mjtNum>()
#define CHECK_MJTNUM_ARRAY(x,ndim) FAIL_IF_FALSE((x).get_dtype() == MJTNUM_DTYPE && (x).get_nd() == (ndim) && (x).get_flags() & bn::ndarray::C_CONTIGUOUS)
// #define ENSURE_MJTNUM_ARRAY(x) FAIL_IF_FALSE( x.get_dtype() ==  )

#define DO_STEP() do {_StepLowLevel(flgDiff, m_u.data(), cscratch[0], f_ctr, pLAM, pDIST, pDBG, s, jac, f_j);} while(0)

#define COPY_STATE(dest) do {mju_copy((dest), m_data->qpos, NQ); mju_copy((dest)+NQ, m_data->qvel, NV); } while(0)
#define INC_BY_STATE(dest) do {mju_add((dest), (dest), m_data->qpos, NQ); mju_add((dest)+NQ, (dest)+NQ, m_data->qvel, NV); } while(0)
#define DEC_BY_STATE(dest) do {mju_sub((dest), (dest), m_data->qpos, NQ); mju_sub((dest)+NQ, (dest)+NQ, m_data->qvel, NV); } while(0)
#define SCALE_INPLACE(p, scl, n) mju_scl((p), (p), (scl), (n))

#define PYPRINT(obj) do{std::string s1 = bp::extract<std::string>((obj).attr("__repr__")()); std::cout << s1 << std::endl;} while(0)


bn::ndarray PyMJCWorld::Step(const bn::ndarray& x, const bn::ndarray& u) {
  CHECK_MJTNUM_ARRAY(x, 1);
  CHECK_MJTNUM_ARRAY(u, 1);
  FAIL_IF_FALSE(u.shape(0) == m_actuatedDims.size());

  mjtNum* xdata = reinterpret_cast<mjtNum*>(x.get_data());
  mjtNum* udata = reinterpret_cast<mjtNum*>(u.get_data());

  _SetState(xdata);
  _SetControl(udata);


  mjtByte flgDiff=1;
  mjtNum *s=0, *jac=0, *f_j=0, *f_ctr=0;
  DO_STEP();

  bn::ndarray out = bn::empty(bp::make_tuple(NQ+NV), bn::dtype::get_builtin<mjtNum>());    
  mjtNum* outdata = reinterpret_cast<mjtNum*>(out.get_data());
  COPY_STATE(outdata);
  return out;
}

bn::ndarray PyMJCWorld::StepMulti(const bn::ndarray& x, const bn::ndarray& u) {
  CHECK_MJTNUM_ARRAY(x, 2);
  CHECK_MJTNUM_ARRAY(u, 2);

  int n_steps = x.shape(0);
  FAIL_IF_FALSE (u.shape(0) == x.shape(0));

  mjtNum* xdata = reinterpret_cast<mjtNum*>(x.get_data());
  mjtNum* udata = reinterpret_cast<mjtNum*>(u.get_data());

  const int NX = NQ + NV, NU = m_actuatedDims.size();

  bn::ndarray out = bn::empty(bp::make_tuple(n_steps,NX), bn::dtype::get_builtin<mjtNum>());    
  mjtNum* outdata = reinterpret_cast<mjtNum*>(out.get_data());


  for (int i=0; i < n_steps; ++i) {
    _SetState(xdata + NX*i);
    _SetControl(udata + NU*i);

    mjtByte flgDiff=1;
    mjtNum *s=0, *jac=0, *f_j=0, *f_ctr=0;
    DO_STEP();
    COPY_STATE(outdata + NX*i);
  }

  return out;
}

bp::tuple PyMJCWorld::StepMulti2(const bn::ndarray& x, const bn::ndarray& u, const bn::ndarray& done) {
  CHECK_MJTNUM_ARRAY(x, 2);
  CHECK_MJTNUM_ARRAY(u, 2);
  
  const int xcols = NQ+NV,
               ucols = m_actuatedDims.size(),
               ycols = NQ+NV,
               fcols=m_model->ndof,
               dcomcols = 3,
               distcols = m_model->nbody,
               kincols = m_model->ndof*7 + m_model->nbody*28;
  int n_steps = x.shape(0);

  // kincols is cdof thru qfrc_bias inclusive

  FAIL_IF_FALSE (u.shape(0) == x.shape(0));
  FAIL_IF_FALSE(u.shape(1) == ucols);
  FAIL_IF_FALSE(x.shape(1) == xcols);


  mjtNum* xdata = reinterpret_cast<mjtNum*>(x.get_data());
  mjtNum* udata = reinterpret_cast<mjtNum*>(u.get_data());

  uint8_t* donedata = reinterpret_cast<uint8_t*>(done.get_data());

  bn::ndarray yarray = bn::zeros(bp::make_tuple(n_steps,ycols), bn::dtype::get_builtin<mjtNum>());
  bn::ndarray farray = bn::zeros(bp::make_tuple(n_steps,fcols), bn::dtype::get_builtin<mjtNum>());
  bn::ndarray dcomarray = bn::zeros(bp::make_tuple(n_steps,dcomcols), bn::dtype::get_builtin<mjtNum>());
  bn::ndarray distarray = bn::zeros(bp::make_tuple(n_steps,distcols), bn::dtype::get_builtin<mjtNum>());
  bn::ndarray kinarray = bn::zeros(bp::make_tuple(n_steps,kincols), bn::dtype::get_builtin<mjtNum>());

  mjtNum* ydata = reinterpret_cast<mjtNum*>(yarray.get_data());
  mjtNum* fdata = reinterpret_cast<mjtNum*>(farray.get_data());
  mjtNum* dcomdata = reinterpret_cast<mjtNum*>(dcomarray.get_data());
  mjtNum* distdata = reinterpret_cast<mjtNum*>(distarray.get_data());
  mjtNum* kindata =  reinterpret_cast<mjtNum*>(kinarray.get_data());

  for (int i=0; i < n_steps; ++i) {

    if (donedata[i]) continue;

    _SetState(xdata + xcols*i);
    _SetControl(udata + ucols*i);

    mjtByte flgDiff=1;
    mjtNum *s=0, *jac=0, *f_j=0, *f_ctr=0;

    mjtNum *plam = fdata + fcols*i; // note mju_copy(plam, d->lc_f, m->ndof);
    mjtNum *pdist = distdata + distcols*i;

    mjtNum comBefore[3];
    mj_kinematics(m_model, m_data);
    _ComputeCOM(comBefore); // XXX extra calculation

    _StepLowLevel(flgDiff, m_u.data(), cscratch[0], f_ctr, plam, pdist, pDBG, s, jac, f_j);
    
    COPY_STATE(ydata + ycols*i); // y 
    mju_copy(kindata + kincols*i, m_data->cdof, kincols);
    // f is handled by steplowlevel
    mjtNum comAfter[3]; // dcom
    mj_kinematics(m_model, m_data);
    _ComputeCOM(comAfter); // XXX extra calculation
    mju_sub(dcomdata + dcomcols*i, comAfter, comBefore, 3); 
    // dist is handled by steplowlevel

  }

  return bp::make_tuple(yarray, farray, dcomarray, distarray, kinarray);
}

void PyMJCWorld::_ComputeCOM(mjtNum* com) {
	 // XXX is idx 0 always the ground?
  mjtNum tot=0;
  com[0] = com[1] = com[2] = tot = 0;
  for(int i=1; i<m_model->nbody; i++ )
  {
    if( m_model->geom_type[i]==mjPLANE ) continue;
    com[0] += m_data->xpos[3*i+0]*m_model->body_mass[i];
    com[1] += m_data->xpos[3*i+1]*m_model->body_mass[i];
    com[2] += m_data->xpos[3*i+2]*m_model->body_mass[i];
    tot += m_model->body_mass[i];
  }
  com[0] /= tot;
  com[1] /= tot;
  com[2] /= tot; 
}

bp::tuple PyMJCWorld::StepJacobian(const bn::ndarray& x, const bn::ndarray& u) {
    CHECK_MJTNUM_ARRAY(x, 1);
    CHECK_MJTNUM_ARRAY(u, 1);
    FAIL_IF_FALSE(u.shape(0) == m_actuatedDims.size());

    mjtNum* xdata = reinterpret_cast<mjtNum*>(x.get_data());
    mjtNum* udata = reinterpret_cast<mjtNum*>(u.get_data());


    for (int i=0; i < m_actuatedDims.size(); ++i) {
      m_u[m_actuatedDims[i]] = udata[i];
    }

    mjtByte flgDiff=1;
    mjtNum *s=0, *jac=0, *f_j=0, *f_ctr=0;


    const int NX = NQ+NV, NU = m_actuatedDims.size();
    bn::ndarray y =    bn::zeros(bp::make_tuple(NX), bn::dtype::get_builtin<mjtNum>());
    bn::ndarray dydx = bn::zeros(bp::make_tuple(NX, NX), bn::dtype::get_builtin<mjtNum>());
    bn::ndarray dydu = bn::zeros(bp::make_tuple(NU, NX), bn::dtype::get_builtin<mjtNum>());


    mjtNum* py = reinterpret_cast<mjtNum*>(y.get_data());
    mjtNum* pdydx = reinterpret_cast<mjtNum*>(dydx.get_data());
    mjtNum* pdydu = reinterpret_cast<mjtNum*>(dydu.get_data());


    mjtNum eps=1e-6;

    _SetState(xdata);
    _SetControl(udata);
    DO_STEP();
    COPY_STATE(py);
    for (int i=0; i < NX; ++i) { // XXX LOL using the fact that qpos and qvel are stored next to each other continguously

      _SetState(xdata);
      m_data->qpos[i] += eps;
      DO_STEP();
      INC_BY_STATE(pdydx+i*NX);

      _SetState(xdata);
      m_data->qpos[i] -= eps;
      DO_STEP();
      DEC_BY_STATE(pdydx+i*NX);

      SCALE_INPLACE(pdydx + i*NX, 1./(2.*eps), NX);      

    }

    for (int i=0; i < NU; ++i) {

      _SetState(xdata);
      m_u[m_actuatedDims[i]] = udata[i] + eps;
      DO_STEP();
      INC_BY_STATE(pdydu + i*NX);

      _SetState(xdata);
      m_u[m_actuatedDims[i]] = udata[i] - eps;
      DO_STEP();
      DEC_BY_STATE(pdydu + i*NX);

      m_u[m_actuatedDims[i]] = udata[i];

      SCALE_INPLACE(pdydu + i*NX, 1./(2.*eps), NX);      

    }

  return bp::make_tuple(y, dydx.transpose(), dydu.transpose());
}

void PyMJCWorld::Plot(const bn::ndarray& x) {
    CHECK_MJTNUM_ARRAY(x, 1);
    if (!m_mjr) _PlotInit();

    mjtNum* xdata = reinterpret_cast<mjtNum*>(x.get_data());
    // mjtNum* udata = reinterpret_cast<mjtNum*>(u.get_data());

    _SetState(xdata);
    mj_kinematics(m_model, m_data);
    mj_global(m_model, m_data);

    m_mjr->Update(m_model, m_data);

}

bn::ndarray PyMJCWorld::GetImage(const bn::ndarray& x) {
  long dims[3]={480,640,4};
  bn::ndarray out = bn::empty(3, dims, bn::dtype::get_builtin<uint8_t>());
  Plot(x);
  m_mjr->SaveImage(out.get_data());
  using bp::_;
  using bp::slice;
  // using bp::slice;
  // return out;
  out = bp::extract<bn::ndarray>(out[bp::make_tuple(slice(_,_,-1),slice(),bp::make_tuple(2,1,0))].attr("copy")());
  return out;
}


void PyMJCWorld::SetActuatedDims(const bp::object& dims) {
  m_u.assign(NV, 0);
  int ndims = bp::len(dims);
  m_actuatedDims.clear();
  for (int i=0; i < ndims; ++i) {
    m_actuatedDims.push_back(bp::extract<int>(dims[i]));
  }
}


void PyMJCWorld::_PlotInit() {
  m_mjr = new mjRender();
  int x        = 0;
  int y        = 0;
  int width    = 640;
  int height   = 480;  
  m_mjr->Open(m_model->name, x, y, width, height);
  m_mjr->Set(m_model, m_data);
}


void PyMJCWorld::_PreProcess() { 

  // Original code:
  mj_solve1(m_model, m_data);
  mj_contactPrepare(m_model, m_data, 0);
  mj_solve2(m_model, m_data);
  m_data->nlim=m_model->nlmax;
  m_data->ncon=m_model->ncmax;




  // for (int i=0; i < sz_pyramid; ++i) m_slcp_betaVec[i] = 0; // siminit.m line 72
}


mjtNum PyMJCWorld::_ContactImpulse(mjtNum* contact_scratch, mjtNum *f_ctr, const mjtNum * /*qvel_next_user*/) {
  mjModel* m=m_model;
  mjData* d = m_data;

  mjtNum dbg, mu;
  int j, stepDbg;

  #if 0
  mjtNum *alpha, *kappaN, *kappaT, *kappaV, *contact_buffer, *P
  int sz_dist = nl+nc;
  #endif

  mjtNum *J, *A, *f, *v, *scratch;
  mjtByte inCloud;
  int nl    = d->nlim;
  int nc    = d->ncon;
  int sz = nl+mjConDim*nc;
  int sz_pyramid = (m_flagPlanar?2:4)*nc+nl;

   //no need to zero - contactPrepare did that already
   //mju_zero(d->lc_f, SZ_F);
   
   switch( m_contactType ) {
      case ContactType_SPRING:
         mju_mulMatVec(d->lc_v0, d->lc_J, d->qvel_next, sz, NV);
         mj_impulseSpring(m, d, m_spring_kspring, m_spring_kdamp, 1, 1, m_spring_scale);
         return 0;
      case ContactType_SPRINGTAU:
      case ContactType_SLCP:
         mju_mulMatVec(d->lc_v0, d->lc_J, d->qvel_next, sz, NV);
     if (m_contactType == ContactType_SPRINGTAU)
      mj_impulseTau(m, d, m_st_tau, m_st_beta);
     else
      mj_impulseTau2(m, d, m_slcp_tau, m_slcp_tau2, m_slcp_beta, m_slcp_maxIter);
     return 0; // This is a little hack - in this formulation, SLCP no longer exists, there is only springtau
         //if( FLG_CONTACT==CF_SPRINGTAU && !ITER_CLOUD ) return 0; //otherwise, continue with slcp
         f       = contact_scratch;
         v       = contact_scratch + sz_pyramid;
         J       = contact_scratch + sz_pyramid*2;
         A       = contact_scratch + sz_pyramid*2 + sz_pyramid*NV;
         scratch = contact_scratch + sz_pyramid*2 + sz_pyramid*NV + sz_pyramid*sz_pyramid;
         mju_mulMatVec(v, J, d->qvel_next, sz_pyramid, NV);
         sz_pyramid = convexify(m, d, m_flagPlanar, f, v, J, A);

         if( m_contactType==ContactType_SLCP ) { //if we came here from springTau, lc_f is already initialized
#ifdef _WIN32
            inCloud= (f_ctr && !_isnan(f_ctr[0]));
#else
            inCloud= (f_ctr && !std::isnan(f_ctr[0]));
#endif
            if (!inCloud) {
               // tol = 1e-15
               dbg=slcp_solve(sz_pyramid, A, v, f, m_slcp_betaVec, 1E-15, m_slcp_maxIter, scratch);
               if( dbg<0 )
                  return dbg;
               if (f_ctr) mju_copy(f_ctr, f, sz_pyramid);
            } else {
               mju_copy(f, f_ctr, sz_pyramid);
            }
         }
         for (j=0;j<m_slcp_iterCloud;j++) {
            mu=-1; //a small regularizing term to deal with rank-deficient A
            stepDbg = slcp_step(sz_pyramid, A, v, f, m_slcp_betaVec, scratch, scratch+sz_pyramid, &mu);
            if( stepDbg<0 ) return stepDbg-0.1234567;
         }
         deconvexify(m, d, m_flagPlanar, f); // assigns into d->lc_f
         
         //test: J*f should be equal to lc_J*lc_f:
         //mju_mulMatTVec(ds,J,f,LOCAL_sz_cl,NV);
         //mju_mulMatTVec(ds+SZ_F,d->lc_J,d->lc_f,SZ_F,NV);
         return dbg;
         
      case ContactType_CONVEX:
        NOTIMPLEMENTED;
        #if 0
         kappaN = contact_scratch;
         kappaT = contact_scratch + sz_dist;
         kappaV = contact_scratch + sz_dist*2;
         alpha  = contact_scratch + sz_dist*3;
         makeKappa(pkappal[0],pkappac[0], kappaN, d->lc_A, nl, nc);
         makeKappa(0         ,pkappac[1], kappaT, d->lc_A, nl, nc);
         makeKappa(pkappal[1],pkappac[2], kappaV, d->lc_A, nl, nc);
#ifdef _WIN32
         inCloud= (f_ctr && !_isnan(f_ctr[0]));
#else
         inCloud= (f_ctr && !std::isnan(f_ctr[0]));
#endif
         
         //initialize:
         mju_mulMatVec(d->lc_v0, d->lc_J, d->qvel_next, sz, NV);
         mj_impulseTau(m, d, TAU_SPRING, BETA_SPRING);
         
         if (FLG_FWD>0) {
            contact_buffer = contact_scratch + sz_dist*3 + sz;
            if (!inCloud) {
               dbg=mj_impulseConvex(d->lc_f, m, d, kappaN, kappaT, kappaV,
                       dmax, amin, amaxl, amaxc, marginF, marginV, 1e-15, MAX_ITER, contact_buffer);
               if( dbg<0 )
                  return dbg;
               if (f_ctr) mju_copy(f_ctr, d->lc_f, sz);
            } else {
               mju_copy(d->lc_f, f_ctr, sz);
            }
            makeAlpha(alpha, d->lc_dist, amin, amaxl, amaxc, dmax, d->lc_A, nl, nc);
            for (j=0;j<ITER_CLOUD;j++) {
               contact_buffer[sz]=0;
               stepDbg = convex_fwdIter(d->lc_f, m, nl, nc,
                       alpha, kappaN, kappaT, kappaV,
                       d->con_friction, d->lc_A, d->lc_v0, d->lc_vmin,
                       marginF, marginV, contact_buffer, contact_buffer+sz);
               if( stepDbg ) return stepDbg+0.7654321;
            }
         } else { //inv dyn
            P              = contact_scratch + sz_dist*3 + sz;
            contact_buffer = contact_scratch + sz_dist*3 + sz*2;
            mju_copy(d->qvel_next, qvel_next_user, NV);
            mju_mulMatVec(d->lc_v0, d->lc_J, d->qvel_next, sz, NV);
            if (!inCloud) {
               dbg=mj_impulseConvexInv(d->lc_f, m, d, kappaN, kappaT, kappaV,
                       dmax, amin, amaxl, amaxc, marginF, marginV, 1e-15, MAX_ITER, contact_buffer);
               if( dbg<0 )
                  return dbg;
               if (f_ctr) mju_copy(f_ctr, d->lc_f, sz);
            } else {
               mju_copy(d->lc_f, f_ctr, sz);
            }
            mju_zero(alpha, sz);
            makeP(P, d->lc_A, alpha, d->lc_v0, d->lc_vmin, nl, nc, kappaV, marginV);
            makeAlpha(alpha, d->lc_dist, amin, amaxl, amaxc, dmax, d->lc_A, nl, nc);
            for (j=0;j<ITER_CLOUD;j++) {
               contact_buffer[sz]=0;
               stepDbg=convex_invIter(d->lc_f, m, nl, nc,
                       alpha, kappaN, kappaT,
                       d->con_friction, d->lc_A, P,
                       marginF, contact_buffer, contact_buffer+sz);
               if( stepDbg ) return stepDbg+0.7654321;
            }
         }
         return dbg;
         #endif
   }

   PRINT_AND_THROW("UNREACHABLE");
   return 0;
}

mjtByte PyMJCWorld::_StepLowLevel(const mjtByte flgDiff, const mjtNum* tu,
        mjtNum* contact_scratch, mjtNum *f_ctr,
        mjtNum *plam, mjtNum *pdist, mjtNum *pdebug, mjtNum *s, 
        mjtNum *jac, mjtNum *f_j) {

  mjModel* m=m_model;
  mjData* d=m_data;
   int i, j, k, n;
   mjtNum *ds = d->scratch;
   mjtNum dbg, total_mass;
   mjtByte needCoM;

   if (flgDiff) {
      // run kinematic computations
      mj_kinematics(m, d);
      mj_global(m, d);
      mj_crb(m, d);
      mj_factorM(m, d);
      if (m_SZ_F) {
         mj_contactPrepare(m, d, (tu?0:2));//(FLG_CONTACT==CF_SPRINGTAU && !ITER_CLOUD));
      }
   } else {
      // global does this, so if we're not running global, we should manually zero it out
      mju_zero(d->qfrc_ext, NV);
      // contactPrepare does this
      mju_zero(d->lc_f, m_SZ_F);
   }
   mj_rne(m, d, 0);
   mj_passive(m, d);
   

   if( s ) {
      needCoM = 0;
      i=0;
      // start with CoM:
      
      // CoM quat space:
      if( FEAT[0] ) i+=4;
      // CoM position:
      for( k=0; k<3; k++ ) {
         if( FEAT[1+k] != 0.0 ) {
            s[i]=d->com[k];
            i++;
            needCoM++;
         }
      }
      // CoM velocities:
      for( k=0; k<6; k++ ) {
         if( FEAT[4+k] != 0.0 ) {
            s[i]=d->cvel[k];
            i++;
         }
      }

      for (j=1;j<NBODY;j++) {
         // quaternion
         if( FEAT[10*j] != 0.0 ) {
            for( k=0; k<4; k++ ) {
               s[i]=d->xquat[4*j+k];
               i++;
            }
         }
         // position
         for( k=0; k<3; k++ ) {
            if( FEAT[10*j+1+k] != 0.0 ) {
         if ( FEAT[10*j+1+k] == 2.0 ) // Use xanchor
                  s[i]=d->xanchor[3*m->body_jnt_adr[j]+k];
         else // Use regular position
          s[i]=d->xpos[3*j+k];
               i++;
            }
         }
         // velocities
         for( k=0; k<6; k++ ) {
            if( FEAT[10*j+4+k] != 0.0 ) {
               s[i]=d->cvel[6*j+k];
               i++;
            }
         }
      }
    //test:
      //if( i!=NS ) s[0]=sqrt((double)-1.0);
      if( jac ) { //need to compute jacs as well
         total_mass=0;
         i=needCoM; //save room for the CoM's jac, which will be created last
         if( FEAT[0] != 0.0 ) i+=3; //save useless room for the CoM's nonexistent quat's jac
         for( j=1; j<NBODY; j++ ) {
            if( FEAT[10*j] != 0.0 ) {
               mj_jac(m, d, 0, ds, 0, j, 1);
               for( k=0; k<3; k++ ) {
                  mju_copy(jac+i*NV, ds+k*NV, NV);
                  i++;
               }
            }
            if( needCoM || FEAT[10*j+1] != 0.0 || FEAT[10*j+2] != 0.0 || FEAT[10*j+3] != 0.0 ) {
               mj_jac(m, d, ds, 0, 0, j, 1);
               for( k=0; k<3; k++ ) {
                  if( FEAT[10*j+1+k] != 0.0 ) {// keep this body's jac:
                     mju_copy(jac+i*NV, ds+k*NV, NV);
                     i++;
                  }
               }
               if( needCoM ) { // building the CoM's jac:
                  n=0;
                  total_mass+=m->body_mass[j];
                  mju_scl(ds, ds, m->body_mass[j], 3*NV);  //scale by body mass
                  for( k=0; k<3; k++ ){
                     if( FEAT[1+k] != 0.0 ) {
                        mju_addTo(jac+n*NV, ds+k*NV, NV);
                        n++;
                     }
                  }
               }
            }
         }
         if( needCoM ) mju_scl(jac, jac, 1/total_mass, needCoM*NV);
      }
   }
   
   if( f_j ) mju_copy(f_j, d->lc_J, (d->nlim+mjConDim*d->ncon)*NV);

   if( pdist )  {
    // store the smallest contact distance for each body.
      // first, set all contact distances to the maximum value.
      for (i = 0; i < NBODY; i++)
      pdist[i] = 1000.0;
    for (j = 0; j < d->ncon; j++)
    {
      // Address of bodies.
      if (d->con_pair[j] >= m->ncpair) continue;
      int b1 = m->geom_body_id[m->pair_geom1[d->con_pair[j]]];
      int b2 = m->geom_body_id[m->pair_geom2[d->con_pair[j]]];
      double pen = d->lc_dist[d->nlim+j];
      if (pdist[b1] > pen)
        pdist[b1] = pen;
      if (pdist[b2] > pen)
        pdist[b2] = pen;
      // Make sure the ground is one of the bodies -- otherwise we need some special handling.
      //if (b1 != 0 && b2 != 0)
      //  mexErrMsgTxt("Returning X distance and detected collision not with the ground -- consider adding some code to handle this!");
    }
      //min2full_dist(d, d->lc_dist, pdist);
      //min2full_xpos(d, d->con_xpos, pdist+SZ_DIST);
   }
   
   if( !tu ) return 0;
   
   if (m_SZ_F && m_flagPrehoc==0) {
      mj_solve2(m, d);
   } 
   else {
      if (m_integrator>0) {
        mju_addTo(d->qfrc_ext, tu, NV);
      }
      mj_solve2(m, d);
   }
   if (m_SZ_F) {
      dbg = _ContactImpulse(contact_scratch, f_ctr, tu); //we give tu to contact because 
   }
   
   if (m_integrator>0) {
      if (m_flagPrehoc==0) {
         mj_backsubM(m, d, d->qacc, tu);
         mju_scl(d->qacc, d->qacc, m->timestep, NV);
         mju_addTo(d->qvel_next, d->qacc, NV);
      }
      mj_solve3(m, d);
      mj_integrate(m, d);
   } else {
      mju_mulMatTVec(d->scratch, d->lc_JP, d->lc_f, m_SZ_F, NV);
      mju_sub(d->qvel_next, tu, d->scratch, NV);
      mju_sub(d->qvel_next, d->qvel_next, d->qvel, NV);
      mju_scl(d->qacc, d->qvel_next, 1/m->timestep, NV);
      mj_fullM(m, d, d->scratch, d->qM);
      mju_mulMatVec(d->qvel_next, d->scratch, d->qacc, NV, NV);
      //first i multiplied by JP, then mult by fullM;
      //alternatively, maybe mult by J and not mult that piece by M?
      mju_sub(d->qvel_next, d->qvel_next, d->qfrc_bias, NV);
      mju_sub(d->qvel_next, d->qvel_next, d->qfrc_ext, NV);
   }
   
   if( plam ) { // return contact forces.
    int sz = d->nlim + mjConDim*d->ncon;
    //mju_mulMatTVec(d->scratch, d->lc_JP, d->lc_f, sz, m->ndof); // This version computes accelerations.
    mju_mulMatTVec(d->scratch, d->lc_J, d->lc_f, sz, m->ndof); // This version computes forces.
    mju_copy(plam, d->scratch, m->ndof);
    //min2full_lam(d, d->lc_f, plam);
   }
   if( pdebug ) pdebug[0]=dbg;
   return 0;
}

bn::ndarray PyMJCWorld::ComputeContacts(const bn::ndarray& x) {


  CHECK_MJTNUM_ARRAY(x, 1);
  mjtNum* xdata = reinterpret_cast<mjtNum*>(x.get_data());
  _SetState(xdata);
  mj_kinematics(m_model, m_data);
  mj_global(m_model, m_data);



  bn::ndarray dists = bn::empty(bp::make_tuple(m_model->nbody), bn::dtype::get_builtin<mjtNum>());    
  mjtNum* distsdata = reinterpret_cast<mjtNum*>(dists.get_data());
  for (int i=0; i < m_model->nbody; ++i) {
    // printf("body %i parent id %i\n", i, m_model->body_parent_id[i]);
    distsdata[i] = 1000;
  }


  // Code copied from mj_contactPrepare
  mjContact con[mjMaxConPair]; 
  mjModel* m=m_model;

  for(int i=0; i<m->ncpair; i++ )
    if( m->pair_enable[i] )
    {
      // get geom ids
      int gid1 = m->pair_geom1[i];
      int gid2 = m->pair_geom2[i];


      int num = conFunction[m->pair_type[i]](con, m->mindist[0],
        m_data->geom_xpos+3*gid1, m_data->geom_xmat+9*gid1, m->geom_size+mjNumSize*gid1,
        m_data->geom_xpos+3*gid2, m_data->geom_xmat+9*gid2, m->geom_size+mjNumSize*gid2);

      if( num > conInfo[m->pair_type[i]][0] )
        mju_error("too many contacts in geom pair");
         
      // process all returned contacts
      for(int j=0; j<num; j++ )
      {
        mjtNum dist = -con[j].depth;
        int bid1 = m_model->geom_body_id[gid1];
        int bid2 = m_model->geom_body_id[gid2];
        distsdata[bid1] = fmin(dist, distsdata[bid1]);
        distsdata[bid2] = fmin(dist, distsdata[bid2]);
      }
    }
  return dists;

}

int _ndarraysize(const bn::ndarray& arr) {
  int prod = 1;
  for (int i=0; i < arr.get_nd(); ++i) {
    prod *= arr.shape(i);
  }
  return prod;
}
template<class T>
void _copyscalardata(const bp::object& from, T& to) {
  to = bp::extract<T>(from);
}
template <short ndim,typename T>
void _copyarraydata(const bn::ndarray& from, T* to) {
  FAIL_IF_FALSE(from.get_dtype() == bn::dtype::get_builtin<T>() && from.get_nd() == ndim && from.get_flags() & bn::ndarray::C_CONTIGUOUS);
  memcpy(to, from.get_data(), _ndarraysize(from)*sizeof(T));
}

template<typename T>
void _csdihk(bp::dict d, const char* key, T& to) {
  // copy scalar data if has_key
  if (d.has_key(key)) _copyscalardata(d[key], to);
}
template<short ndim, typename T>
void _cadihk(bp::dict d, const char* key, T* to) {
  // copy array data if has_key
  if (d.has_key(key)) {
    bn::ndarray arr = bp::extract<bn::ndarray>(d[key]);
    _copyarraydata<ndim,T>(arr, to);
  }
}

void PyMJCWorld::SetModel(bp::dict d) {
  _cadihk<1>(d, "dof_armature", m_model->dof_armature);
  _cadihk<2>(d, "jnt_limit", m_model->jnt_limit);
}

bp::dict PyMJCWorld::GetModel() {
  bp::dict out;
  out["nqpos"] = m_model->nqpos;
  out["ndof"] = m_model->ndof;
  out["njnt"] = m_model->njnt;
  out["nbody"] = m_model->nbody;
  out["ngeom"] = m_model->ngeom;
  out["neqmax"] = m_model->neqmax;
  out["nlmax"] = m_model->nlmax;
  out["ncpair"] = m_model->ncpair;
  out["ncmax"] = m_model->ncmax;
  out["name"] = bp::str(m_model->name);
  out["timestep"] = m_model->timestep;
  out["gravity"] = toNdarray1<mjtNum>(m_model->gravity,3);
  out["viscosity"] = m_model->viscosity;
  out["mindist"] = toNdarray1<mjtNum>(m_model->mindist,2);
  out["erreduce"] = toNdarray1<mjtNum>(m_model->errreduce, 3);

  out["body_mass"] = toNdarray1<mjtNum>(m_model->body_mass, m_model->nbody);
  out["body_inertia"] = toNdarray2<mjtNum>(m_model->body_inertia, m_model->nbody,3);
  out["body_pos"] = toNdarray2<mjtNum>(m_model->body_pos, m_model->nbody,3);
  out["body_quat"] = toNdarray2<mjtNum>(m_model->body_quat, m_model->nbody,4);
  out["body_viscoef"] = toNdarray1<mjtNum>(m_model->body_viscoef, m_model->nbody);
  out["body_parent_id"] = toNdarray1<int>(m_model->body_parent_id, m_model->nbody);
  out["body_jnt_num"] = toNdarray1<int>(m_model->body_jnt_num, m_model->nbody);
  out["body_dof_num"] = toNdarray1<int>(m_model->body_dof_num, m_model->nbody);
  out["body_root_id"] = toNdarray1<int>(m_model->body_root_id, m_model->nbody);

  out["jnt_type"] = toNdarray1<int>((int*)m_model->jnt_type, m_model->njnt);
  out["jnt_pos"] = toNdarray2<mjtNum>(m_model->jnt_pos, m_model->njnt,3);
  out["jnt_axis"] = toNdarray2<mjtNum>(m_model->jnt_axis, m_model->njnt,3);
  out["jnt_spring"] = toNdarray1<mjtNum>(m_model->jnt_spring, m_model->njnt);
  out["jnt_damping"] = toNdarray1<mjtNum>(m_model->jnt_damping, m_model->njnt);
  out["jnt_vel_damping"] = toNdarray1<mjtNum>(m_model->jnt_vel_damping, m_model->njnt);
  out["jnt_body_id"] = toNdarray1<int>(m_model->jnt_body_id, m_model->njnt);
  out["jnt_limit"] = toNdarray2<mjtNum>(m_model->jnt_limit, m_model->njnt,2);
  out["jnt_islimited"] = toNdarray1<mjtByte>(m_model->jnt_islimited, m_model->njnt);

  out["dof_armature"] = toNdarray1<mjtNum>(m_model->dof_armature, m_model->ndof);
  out["dof_body_id"] = toNdarray1<int>(m_model->dof_body_id, m_model->ndof);
  out["dof_jnt_id"] = toNdarray1<int>(m_model->dof_jnt_id, m_model->ndof);
  out["dof_parent_id"] = toNdarray1<int>(m_model->dof_parent_id, m_model->ndof);

  out["geom_type"] = toNdarray1<int>((int*)m_model->geom_type, m_model->ngeom);
  out["geom_size"] = toNdarray2<mjtNum>(m_model->geom_size, m_model->ngeom, mjNumSize);
  out["geom_pos"] = toNdarray2<mjtNum>(m_model->geom_pos, m_model->ngeom,3);
  out["geom_quat"] = toNdarray2<mjtNum>(m_model->geom_quat, m_model->ngeom,4);
  out["geom_color"] = toNdarray2<mjtNum>(m_model->geom_color, m_model->ngeom,3);
  out["geom_body_id"] = toNdarray1<int>(m_model->geom_body_id, m_model->ngeom);
  out["geom_isoffset"] = toNdarray1<mjtByte>(m_model->geom_isoffset, m_model->ngeom);

  out["eq_body1"] = toNdarray1<int>(m_model->eq_body1, m_model->neqmax);
  out["eq_body2"] = toNdarray1<int>(m_model->eq_body2, m_model->neqmax);
  out["eq_pos1"] = toNdarray2<mjtNum>(m_model->eq_pos1, m_model->neqmax, 3);
  out["eq_pos2"] = toNdarray2<mjtNum>(m_model->eq_pos2, m_model->neqmax, 3);
  out["eq_enable"] = toNdarray1<mjtByte>(m_model->eq_enable, m_model->neqmax);

  out["pair_type"] = toNdarray1<int>((int*)m_model->pair_type, m_model->ncpair);
  out["pair_friction"] = toNdarray1<mjtNum>(m_model->pair_friction, m_model->ncpair);
  out["pair_geom1"] = toNdarray1<int>(m_model->pair_geom1, m_model->ncpair);
  out["pair_geom2"] = toNdarray1<int>(m_model->pair_geom2, m_model->ncpair);
  out["pair_enable"] = toNdarray1<mjtByte>(m_model->pair_enable, m_model->ncpair);

  return out;
}

bp::dict PyMJCWorld::GetData(const bn::ndarray& x) {
  NOTIMPLEMENTED;
}

BOOST_PYTHON_MODULE(mjcpy) {
    bn::initialize();

    bp::enum_<ContactType>("ContactType")
      .value("NONE",ContactType_SPRING)
      .value("SLCP",ContactType_SLCP)
      .value("SPRING",ContactType_SPRINGTAU)
      .value("CONVEX",ContactType_CONVEX)
      ;

    bp::class_<PyMJCWorld,boost::noncopyable>("MJCWorld","docstring here", bp::init<const std::string&>())

        .def("Step",&PyMJCWorld::Step)
        .def("StepMulti",&PyMJCWorld::StepMulti)
        .def("StepMulti2",&PyMJCWorld::StepMulti2)
        .def("StepJacobian", &PyMJCWorld::StepJacobian)
        .def("Plot",&PyMJCWorld::Plot)
        .def("SetActuatedDims",&PyMJCWorld::SetActuatedDims)
        .def("ComputeContacts", &PyMJCWorld::ComputeContacts)
        .def("SetTimestep",&PyMJCWorld::SetTimestep)
        .def("SetContactType",&PyMJCWorld::SetContactType)
        .def("GetModel",&PyMJCWorld::GetModel)
        .def("SetModel",&PyMJCWorld::SetModel)
        .def("GetData",&PyMJCWorld::GetData)
        .def("GetImage",&PyMJCWorld::GetImage)
        ;
}
