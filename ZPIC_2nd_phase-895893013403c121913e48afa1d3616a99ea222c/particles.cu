/*
 * particles.cu
 * Implementação CUDA ZPIC - Versão Final Limpa (Atomics + Physics)
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "particles.h"
#include "random.h"
#include "emf.h"
#include "current.h"
#include "zdf.h"
#include "timer.h"

// Compatibilidade C++
#define restrict __restrict__

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Erro CUDA: %s:%d, ", __FILE__, __LINE__); \
        printf("Code:%d, Reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

static double _spec_time = 0.0;
static uint64_t _spec_npush = 0;

void spec_sort( t_species *spec );
void spec_move_window( t_species *spec );

// =========================================================
// FUNÇÕES AUXILIARES NA GPU (__device__)
// =========================================================

__device__ inline int d_ltrim( float x )
{
    return ( x >= 1.0f ) - ( x < 0.0f );
}

__device__ inline void d_interpolate_fld( const float3* __restrict__ E, 
                                          const float3* __restrict__ B,
                                          const t_part* __restrict__ part, 
                                          float3* __restrict__ Ep, 
                                          float3* __restrict__ Bp )
{
    int i, ih;
    float w1, w1h;

    i = part->ix;
    w1 = part->x;
    
    ih = (w1 < 0.5f) ? -1 : 0;
    w1h = w1 + ((w1 < 0.5f) ? 0.5f : -0.5f);
    ih += i;

    Ep->x = E[ih].x * (1.0f - w1h) + E[ih+1].x * w1h;
    Ep->y = E[i ].y * (1.0f -  w1) + E[i+1 ].y * w1;
    Ep->z = E[i ].z * (1.0f -  w1) + E[i+1 ].z * w1;

    Bp->x = B[i ].x * (1.0f  - w1) + B[i+1 ].x * w1;
    Bp->y = B[ih].y * (1.0f - w1h) + B[ih+1].y * w1h;
    Bp->z = B[ih].z * (1.0f - w1h) + B[ih+1].z * w1h;
}

// ---------------------------------------------------------
// NOVA FUNÇÃO: DEPOSIÇÃO DE CORRENTE COM ATOMICS
// ---------------------------------------------------------
__device__ void d_dep_current_zamb( int ix0, int di, float x0, float dx, 
                                    float qnx, float qvy, float qvz, 
                                    float3* __restrict__ J )
{
    struct t_vp {
        float x0, x1, dx, qvy, qvz;
        int ix;
    } vp[2]; 

    int vnp = 1;

    vp[0].x0 = x0;
    vp[0].dx = dx;
    vp[0].x1 = x0 + dx;
    vp[0].qvy = qvy * 0.5f;
    vp[0].qvz = qvz * 0.5f;
    vp[0].ix = ix0;

    if ( di != 0 ) {
        int ib = ( di == 1 );
        float delta = (x0 + dx - ib) / dx;

        vp[1].x0 = 1.0f - ib;
        vp[1].x1 = (x0 + dx) - di;
        vp[1].dx = dx * delta;
        vp[1].ix = ix0 + di;
        vp[1].qvy = vp[0].qvy * delta;
        vp[1].qvz = vp[0].qvz * delta;

        vp[0].x1 = ib;
        vp[0].dx *= (1.0f - delta);
        vp[0].qvy *= (1.0f - delta);
        vp[0].qvz *= (1.0f - delta);

        vnp++;
    }

    for (int k = 0; k < vnp; k++) {
        float S0x[2], S1x[2];
        
        // VARIÁVEIS INÚTEIS REMOVIDAS AQUI

        S0x[0] = 1.0f - vp[k].x0; 
        S0x[1] = vp[k].x0;
        S1x[0] = 1.0f - vp[k].x1; 
        S1x[1] = vp[k].x1;

        float w_base = S0x[0] + S1x[0] + (S0x[0] - S1x[0]) * 0.5f;
        float w_next = S0x[1] + S1x[1] + (S0x[1] - S1x[1]) * 0.5f;

        // ATOMIC ADD
        atomicAdd( &J[ vp[k].ix ].x, qnx * vp[k].dx );
        
        atomicAdd( &J[ vp[k].ix ].y, vp[k].qvy * w_base );
        atomicAdd( &J[ vp[k].ix + 1 ].y, vp[k].qvy * w_next );

        atomicAdd( &J[ vp[k].ix ].z, vp[k].qvz * w_base );
        atomicAdd( &J[ vp[k].ix + 1 ].z, vp[k].qvz * w_next );
    }
}

// =========================================================
// KERNEL CUDA
// =========================================================
__global__ void k_spec_advance( 
    t_part* part,         
    int np,               
    float dt, float dx, 
    float m_q, float q,
    const float3* E_grid, 
    const float3* B_grid, 
    float3* J_grid,       
    int nx )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= np) return;

    const float tem   = 0.5f * dt / m_q;
    const float dt_dx = dt / dx;
    const float qnx   = q * dx / dt; 

    float3 Ep, Bp;
    float utx, uty, utz;
    float ux, uy, uz, u2;
    float gamma, rg, gtem, otsq;
    float x1, dx_move;
    int di;

    ux = part[i].ux;
    uy = part[i].uy;
    uz = part[i].uz;

    d_interpolate_fld( E_grid, B_grid, &part[i], &Ep, &Bp );

    Ep.x *= tem; Ep.y *= tem; Ep.z *= tem;

    utx = ux + Ep.x;
    uty = uy + Ep.y;
    utz = uz + Ep.z;

    u2 = utx*utx + uty*uty + utz*utz;
    gamma = sqrtf( 1.0f + u2 );
    gtem = tem / gamma;

    Bp.x *= gtem; Bp.y *= gtem; Bp.z *= gtem;
    otsq = 2.0f / ( 1.0f + Bp.x*Bp.x + Bp.y*Bp.y + Bp.z*Bp.z );

    ux = utx + uty*Bp.z - utz*Bp.y;
    uy = uty + utz*Bp.x - utx*Bp.z;
    uz = utz + utx*Bp.y - uty*Bp.x;

    Bp.x *= otsq; Bp.y *= otsq; Bp.z *= otsq;

    utx += uy*Bp.z - uz*Bp.y;
    uty += uz*Bp.x - ux*Bp.z;
    utz += ux*Bp.y - uy*Bp.x;

    ux = utx + Ep.x;
    uy = uty + Ep.y;
    uz = utz + Ep.z;

    part[i].ux = ux;
    part[i].uy = uy;
    part[i].uz = uz;

    rg = 1.0f / sqrtf(1.0f + ux*ux + uy*uy + uz*uz);
    dx_move = dt_dx * rg * ux;
    x1 = part[i].x + dx_move;

    di = d_ltrim(x1);
    
    // Calcular correntes
    float qvy = q * uy * rg;
    float qvz = q * uz * rg;

    d_dep_current_zamb( part[i].ix, di, part[i].x, dx_move, qnx, qvy, qvz, J_grid );

    x1 -= di;
    part[i].x = x1;
    part[i].ix += di;
}

// =========================================================
// FUNÇÕES DE ESTATÍSTICA
// =========================================================

double spec_time( void ) { return _spec_time; }
uint64_t spec_npush( void ) { return _spec_npush; }
double spec_perf( void ) { return (_spec_npush > 0 )? _spec_time / _spec_npush: -1.0; }

// =========================================================
// FUNÇÕES DE INICIALIZAÇÃO (CPU)
// =========================================================

void spec_set_u( t_species* spec, const int start, const int end )
{
    for (int i = start; i <= end; i++) {
        spec->part[i].ux = spec -> uth[0] * rand_norm();
        spec->part[i].uy = spec -> uth[1] * rand_norm();
        spec->part[i].uz = spec -> uth[2] * rand_norm();
    }
    float3 * restrict net_u = (float3 *) malloc( spec->nx * sizeof(float3));
    int * restrict    npc   = (int *) malloc( spec->nx * sizeof(int));
    memset(net_u, 0, spec->nx * sizeof(float3) );
    memset(npc, 0, (spec->nx) * sizeof(int) );
    for (int i = start; i <= end; i++) {
        const int idx  = spec -> part[i].ix;
        net_u[ idx ].x += spec->part[i].ux;
        net_u[ idx ].y += spec->part[i].uy;
        net_u[ idx ].z += spec->part[i].uz;
        npc[ idx ] += 1;
    }
    for(int i =0; i< spec->nx; i++ ) {
        const float norm = (npc[ i ] > 0) ? 1.0f/npc[i] : 0;
        net_u[ i ].x *= norm;
        net_u[ i ].y *= norm;
        net_u[ i ].z *= norm;
    }
    for (int i = start; i <= end; i++) {
        const int idx  = spec -> part[i].ix;
        spec->part[i].ux += spec -> ufl[0] - net_u[ idx ].x;
        spec->part[i].uy += spec -> ufl[1] - net_u[ idx ].y;
        spec->part[i].uz += spec -> ufl[2] - net_u[ idx ].z;
    }
    free( npc ); free( net_u );
}

int spec_np_inj( t_species* spec, const int range[] )
{
    int np_inj;
    switch ( spec -> density.type ) {
    case STEP: 
        {
            int i0 = spec -> density.start / spec -> dx - spec -> n_move;
            if ( i0 > range[1] ) np_inj = 0;
            else {
                if ( i0 < range[0] ) i0 = range[0];
                np_inj = ( range[1] - i0 + 1 ) * spec -> ppc;
            }
        }
        break;
    case SLAB: 
        {
            int i0 = spec -> density.start / spec -> dx - spec -> n_move;
            int i1 = spec -> density.end / spec -> dx - spec -> n_move;
            if ( (i0 > range[1]) || (i1 < range[0]) ) np_inj = 0;
            else {
                if ( i0 < range[0] ) i0 = range[0];
                if ( i1 > range[1] ) i1 = range[1];
                np_inj = ( i1 - i0 + 1 ) * spec -> ppc;
            }
        }
        break;
    case RAMP: 
        {
            float x0 = spec -> density.start;
            float x1 = spec -> density.end;
            float a = (range[0] + spec -> n_move) * spec->dx;
            float b = (range[1] + 1 + spec -> n_move) * spec->dx;
            if ( (x1 <= x0) || (a > x1) || (b < x0) ) np_inj = 0;
            else {
                if ( a < x0 ) a = x0;
                if ( b > x1 ) b = x1;
                float n0 = spec -> density.ramp[0];
                float n1 = spec -> density.ramp[1];
                float q = (b-a)*( n0 + 0.5 * (a+b-2*x0)*(n1-n0)/(x1-x0));
                np_inj = q * spec -> ppc / spec -> dx;
            }
        }
        break;
    case CUSTOM: 
        {
            double q = 0.5 * ( (*spec -> density.custom)((range[0] + spec -> n_move) * spec->dx,
                                                         spec -> density.custom_data) +
                               (*spec -> density.custom)((range[1] + 1 + spec -> n_move) * spec->dx,
                                                         spec -> density.custom_data) );
            for( int i = range[0]+1; i <= range[1]; i++) {
                q += (*spec -> density.custom)((i + spec -> n_move) * spec->dx,
                                               spec -> density.custom_data);
            }
            np_inj = ceil(q * spec -> ppc);
        }
        break;
    case EMPTY: np_inj = 0; break;
    default: np_inj = ( range[1] - range[0] + 1 ) * spec -> ppc;
    }
    return np_inj;
}

void spec_set_x( t_species* spec, const int range[] )
{
    int i, k, ip;
    float start, end;
    const int npc = spec->ppc;
    float* poscell = (float*) malloc(npc * sizeof(float));
    for (i=0; i<spec->ppc; i++) poscell[i] = ( i + 0.5 ) / npc;
    ip = spec -> np;
    switch ( spec -> density.type ) {
    case STEP: 
        start = spec -> density.start / spec -> dx - spec -> n_move;
        for (i = range[0]; i <= range[1]; i++) {
            for (k=0; k<npc; k++) {
                if ( i + poscell[k] > start ) {
                    spec->part[ip].ix = i; spec->part[ip].x = poscell[k]; ip++;
                }
            }
        }
        break;
    case SLAB: 
        start = spec -> density.start / spec -> dx - spec -> n_move;
        end   = spec -> density.end / spec -> dx - spec -> n_move;
        for (i = range[0]; i <= range[1]; i++) {
            for (k=0; k<npc; k++) {
                if ( i + poscell[k] > start &&  i + poscell[k] < end ) {
                    spec->part[ip].ix = i; spec->part[ip].x = poscell[k]; ip++;
                }
            }
        }
        break;
    case RAMP: 
        {
            double r0 = spec -> density.start / spec -> dx;
            double r1 = spec -> density.end / spec -> dx;
            if (((range[0] + spec -> n_move) > r1 ) || ((range[1] + spec -> n_move) < r0 )) break;
            double n0 = spec -> density.ramp[0];
            double n1 = spec -> density.ramp[1];
            if ( r0 < 0 ) { n0 += - r0 * (n1-n0) / (r1-r0); r0 = 0; }
            double cpp = 1.0 / spec->ppc;
            for( k = spec -> density.total_np_inj; ; k++ ) {
                double Rs = (k+0.5) * cpp / (r1 - r0);
                double pos = 2 * Rs / (sqrt( n0*n0 + 2 * (n1-n0) * Rs ) + n0);
                if ( pos > 1 ) break;
                pos = r0 + (r1-r0) * pos;
                int ix = pos;
                if ( ix - spec -> n_move < range[0] ) break;
                if ( ix - spec -> n_move > range[1] ) break;
                spec->part[ip].ix = ix - spec -> n_move; spec->part[ip].x = pos - ix; ip++;
            }
        }
        break;
    case CUSTOM: 
        {
            const double dx = spec -> dx;
            const double cpp = 1.0 / spec->ppc;
            k = spec -> density.total_np_inj;
            int ix = range[0];
            double n0;
            double n1 = (*spec -> density.custom)((ix + spec -> n_move) * dx, spec -> density.custom_data);
            double d0;
            double d1 = spec -> density.custom_q_inj;
            double Rs;
            while( ix <= range[1] ){
                n0 = n1; n1 = (*spec -> density.custom)((ix + 1 + spec -> n_move)*dx, spec -> density.custom_data);
                d0 = d1; d1 += 0.5 * (n0+n1);
                while( ( Rs =  (k+0.5) * cpp ) < d1 ) {
                    double pos = 2 * (Rs-d0) /( sqrt( n0*n0 + 2 * (n1-n0) * (Rs-d0) ) + n0 );
                    spec->part[ip].ix = ix; spec->part[ip].x = pos; ip++; k++;
                }
                ix++;
            }
            spec -> density.custom_q_inj = d1;
        }
        break;
    case EMPTY: break;
    default: // Uniform density
        for (i = range[0]; i <= range[1]; i++) {
            for (k=0; k<npc; k++) {
                spec->part[ip].ix = i; spec->part[ip].x = poscell[k]; ip++;
            }
        }
    }
    spec -> density.total_np_inj += ip - spec -> np;
    spec -> np = ip;
    free(poscell);
}

void spec_grow_buffer( t_species* spec, const int size ) {
    if ( size > spec -> np_max ) {
        spec -> np_max = ( size/1024 + 1) * 1024;
        spec -> part = (t_part*) realloc( (void*) spec -> part, spec -> np_max * sizeof(t_part) );
    }
}

void spec_inject_particles( t_species* spec, const int range[] )
{
    int start = spec -> np;
    int np_inj = spec_np_inj( spec, range );
    spec_grow_buffer( spec, spec -> np + np_inj );
    spec_set_x( spec, range );
    spec_set_u( spec, start, spec -> np - 1 );
}

void spec_new( t_species* spec, char name[], const float m_q, const int ppc,
              const float *ufl, const float * uth,
              const int nx, float box, const float dt, t_density* density )
{
    int i, npc;
    strncpy( spec -> name, name, MAX_SPNAME_LEN );
    spec->nx = nx; spec->ppc = ppc; npc = ppc;
    spec->box = box; spec->dx = box / nx;
    spec -> m_q = m_q; spec -> q = copysign( 1.0f, m_q ) / npc;
    spec -> dt = dt; spec->np_max = 0; spec->part = NULL;

    if ( density ) {
        spec -> density = *density;
        if ( spec -> density.n == 0. ) spec -> density.n = 1.0;
    } else { spec -> density.type = UNIFORM; spec -> density.n = 1.0; }
    spec -> density.total_np_inj = 0; spec -> density.custom_q_inj = 0.;
    spec ->q *= fabsf( spec -> density.n );

    if ( ufl ) { for(i=0; i<3; i++) spec -> ufl[i] = ufl[i]; } 
    else { for(i=0; i<3; i++) spec -> ufl[i] = 0; }
    if ( uth ) { for(i=0; i<3; i++) spec -> uth[i] = uth[i]; } 
    else { for(i=0; i<3; i++) spec -> uth[i] = 0; }

    spec -> iter = 0; spec -> moving_window = 0; spec -> n_move = 0; spec -> np = 0;
    const int range[2] = {0, nx-1};
    spec_inject_particles( spec, range );
    spec -> n_sort = 100000; spec -> bc_type = PART_BC_PERIODIC;
}

void spec_move_window( t_species *spec ){
    if ((spec->iter * spec->dt ) > (spec->dx * (spec->n_move + 1)))  {
        int i;
        for( i = 0; i < spec->np; i++ ) spec->part[i].ix--;
        spec -> n_move++;
        const int range[2] = {spec->nx-1,spec->nx-1};
        spec_inject_particles( spec, range );
    }
}

void spec_delete( t_species* spec )
{
    free(spec->part);
    spec->np = -1;
}

// =========================================================
// DEPOSIÇÃO DE CORRENTE (CPU)
// =========================================================

void dep_current_esk( int ix0, int di, float x0, float x1, float qnx, float qvy, float qvz, t_current *current )
{
    float S0x[4], S1x[4], DSx[4], Wx[4], Wy[4], Wz[4];

    S0x[0] = 0.0f; S0x[1] = 1.0f - x0; S0x[2] = x0; S0x[3] = 0.0f;
    for (int i=0; i<4; i++) { S1x[i] = 0.0f; }
    S1x[ 1 + di ] = 1.0f - x1;
    S1x[ 2 + di ] = x1;
    for (int i=0; i<4; i++) { DSx[i] = S1x[i] - S0x[i]; }
    for (int i=0; i<4; i++) {
        Wx[i] = qnx * DSx[i];
        Wy[i] = qvy * (S0x[i] + DSx[i]/2.0f);
        Wz[i] = qvz * (S0x[i] + DSx[i]/2.0f);
    }
    float3* restrict const J = current -> J;
    float c = - Wx[0];
    J[ ix0 - 1 ].x += c;
    for (int i=1; i<4; i++) {
        c -=  Wx[i];
        J[ ix0 + i ].x += c;
    }
    for (int i=0; i<4; i++) {
        J[ ix0 + i - 1 ].y += Wy[ i ];
        J[ ix0 + i - 1 ].z += Wz[ i ];
    }
}

void dep_current_zamb( int ix0, int di, float x0, float dx, float qnx, float qvy, float qvz, t_current *current )
{
    typedef struct {
        float x0, x1, dx, qvy, qvz;
        int ix;
    } t_vp;
    t_vp vp[3];
    int vnp = 1;

    vp[0].x0 = x0; vp[0].dx = dx; vp[0].x1 = x0+dx;
    vp[0].qvy = qvy/2.0; vp[0].qvz = qvz/2.0; vp[0].ix = ix0;

    if ( di != 0 ) {
        int ib = ( di == 1 );
        float delta = (x0+dx-ib)/dx;
        vp[1].x0 = 1-ib; vp[1].x1 = (x0 + dx) - di; vp[1].dx = dx*delta; vp[1].ix = ix0 + di;
        vp[1].qvy = vp[0].qvy*delta; vp[1].qvz = vp[0].qvz*delta;
        vp[0].x1 = ib; vp[0].dx *= (1.0f-delta);
        vp[0].qvy *= (1.0f-delta); vp[0].qvz *= (1.0f-delta);
        vnp++;
    }
    float3* restrict const J = current -> J;
    for (int k = 0; k < vnp; k++) {
        float S0x[2], S1x[2];
        S0x[0] = 1.0f - vp[k].x0; S0x[1] = vp[k].x0;
        S1x[0] = 1.0f - vp[k].x1; S1x[1] = vp[k].x1;
        J[ vp[k].ix     ].x += qnx * vp[k].dx;
        J[ vp[k].ix     ].y += vp[k].qvy * (S0x[0]+S1x[0]+(S0x[0]-S1x[0])/2.0f);
        J[ vp[k].ix + 1 ].y += vp[k].qvy * (S0x[1]+S1x[1]+(S0x[1]-S1x[1])/2.0f);
        J[ vp[k].ix     ].z += vp[k].qvz * (S0x[0]+S1x[0]+(S0x[0]-S1x[0])/2.0f);
        J[ vp[k].ix  +1 ].z += vp[k].qvz * (S0x[1]+S1x[1]+(S0x[1]-S1x[1])/2.0f);
    }
}

// =========================================================
// SORTING (CPU)
// =========================================================
void spec_sort( t_species* spec )
{
    const int ncell = spec->nx;
    int * restrict idx  = (int *) malloc(spec->np*sizeof(int));
    int * restrict npic = (int *) malloc( ncell * sizeof(int));
    memset( npic, 0, ncell * sizeof(int));
    for (int i=0; i<spec->np; i++) { idx[i] = spec->part[i].ix; npic[idx[i]]++; }
    int isum = 0;
    for (int i=0; i<ncell; i++) { int j = npic[i]; npic[i] = isum; isum += j; }
    for (int i=0; i< spec->np; i++) { int j = idx[i]; idx[i] = npic[j]++; }
    free(npic);
    for (int i=0; i < spec->np; i++) {
        int k = idx[i];
        while ( k > i ) {
            t_part tmp = spec->part[k];
            spec->part[k] = spec->part[i];
            spec->part[i] = tmp;
            int t = idx[k]; idx[k] = -1; k = t;
        }
    }
    free(idx);
}

// =========================================================
// FUNÇÃO PRINCIPAL DE AVANÇO (GPU + CPU VERIFICATION)
// =========================================================

void spec_advance( t_species* spec, t_emf* emf, t_current* current )
{
    uint64_t t0 = timer_ticks();

    t_part *d_part;
    float3 *d_E, *d_B, *d_J;

    size_t size_part = spec->np * sizeof(t_part);
    size_t size_grid = (spec->nx + 2) * sizeof(float3); 

    // Alocar na GPU
    CHECK_CUDA(cudaMalloc((void**)&d_part, size_part));
    CHECK_CUDA(cudaMalloc((void**)&d_E, size_grid));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size_grid));
    CHECK_CUDA(cudaMalloc((void**)&d_J, size_grid));

    // Copiar dados para GPU
    CHECK_CUDA(cudaMemcpy(d_part, spec->part, size_part, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, emf->E_part, size_grid, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, emf->B_part, size_grid, cudaMemcpyHostToDevice));
    
    // IMPORTANTE: Limpar o buffer de corrente na GPU (memset a 0)
    CHECK_CUDA(cudaMemset(d_J, 0, size_grid));

    // Lançar Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (spec->np + threadsPerBlock - 1) / threadsPerBlock;

    k_spec_advance<<<blocksPerGrid, threadsPerBlock>>>(
        d_part,
        spec->np,
        spec->dt, spec->dx,
        spec->m_q, spec->q,
        d_E, d_B, d_J, // Passar o buffer de corrente
        spec->nx
    );

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // Recuperar Partículas
    CHECK_CUDA(cudaMemcpy(spec->part, d_part, size_part, cudaMemcpyDeviceToHost));
    
    // Recuperar Corrente (Somar ao buffer do CPU)
    float3* J_host_temp = (float3*) malloc(size_grid);
    CHECK_CUDA(cudaMemcpy(J_host_temp, d_J, size_grid, cudaMemcpyDeviceToHost));
    
    // Como zpic pode ter múltiplas espécies, temos de somar ao current->J em vez de substituir
    for(int i=0; i<spec->nx+2; i++){
        current->J[i].x += J_host_temp[i].x;
        current->J[i].y += J_host_temp[i].y;
        current->J[i].z += J_host_temp[i].z;
    }
    free(J_host_temp);

    // Libertar GPU
    CHECK_CUDA(cudaFree(d_part)); CHECK_CUDA(cudaFree(d_E)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_J));

    // --- CÁLCULO DE ENERGIA (CPU) ---
    double energy = 0;
    for (int i = 0; i < spec->np; i++) {
        float ux = spec->part[i].ux;
        float uy = spec->part[i].uy;
        float uz = spec->part[i].uz;
        float u2 = ux*ux + uy*uy + uz*uz;
        float gamma = sqrtf(1.0f + u2);
        energy += u2 / (1.0f + gamma);
    }
    spec->energy = spec->q * spec->m_q * energy * spec->dx;

    spec -> iter += 1;

    // Boundary conditions
    if ( spec -> moving_window || spec -> bc_type == PART_BC_OPEN ){
        if (spec -> moving_window ) spec_move_window( spec );
        int i = 0;
        while ( i < spec -> np ) {
            if (( spec -> part[i].ix < 0 ) || ( spec -> part[i].ix >= spec->nx )) {
                spec -> part[i] = spec -> part[ -- spec -> np ];
                continue;
            }
            i++;
        }
    } else {
        for (int i=0; i<spec->np; i++) {
            spec -> part[i].ix += (( spec -> part[i].ix < 0 ) ? spec->nx : 0 ) - (( spec -> part[i].ix >= spec->nx ) ? spec->nx : 0);
        }
    }

    if ( spec -> n_sort > 0 ) {
        if ( ! (spec -> iter % spec -> n_sort) ) spec_sort( spec );
    }

    _spec_npush += spec -> np;
    _spec_time += timer_interval_seconds( t0, timer_ticks() );
}

// =========================================================
// DIAGNÓSTICOS (MANTIDOS)
// =========================================================

void spec_deposit_charge( const t_species* spec, float* charge )
{
    const float q = spec -> q;
    for (int i=0; i<spec->np; i++) {
        int idx = spec->part[i].ix;
        float w1 = spec->part[i].x;
        charge[ idx              ] += ( 1.0f - w1 ) * q;
        charge[ idx + 1          ] += (        w1 ) * q;
    }
    if ( ! spec -> moving_window ){
        charge[ 0 ] += charge[ spec -> nx ];
    }
}

void spec_rep_particles( const t_species *spec )
{
    t_zdf_file part_file;
    int i;
    const char * quants[] = { "x", "ux","uy","uz" };
    const char * qlabels[] = { "x", "u_x","u_y","u_z" };
    const char * qunits[] = { "c/\\omega_p", "c","c","c" };

    t_zdf_iteration iter;
    iter.name = (char*)"ITERATION"; iter.n = spec->iter; iter.t = spec -> iter * spec -> dt; iter.time_units = (char*)"1/\\omega_p";

    t_zdf_part_info info;
    info.name = (char *) spec -> name; info.label = (char *) spec -> name;
    info.nquants = 4; info.quants = (char **) quants; info.qlabels = (char **) qlabels; info.qunits = (char **) qunits;
    info.np = spec ->np;

    char path[1024];
    snprintf(path, 1024, "PARTICLES/%s", spec -> name );
    zdf_open_part_file( &part_file, &info, &iter, path );

    size_t size = ( spec -> np ) * sizeof( float );
    float* data = (float*) malloc( size );

    for( i = 0; i < spec ->np; i++ ) data[i] = (spec -> n_move + spec -> part[i].ix + spec -> part[i].x ) * spec -> dx;
    zdf_add_quant_part_file( &part_file, (char*)quants[0], data, spec ->np );

    for( i = 0; i < spec ->np; i++ ) data[i] = spec -> part[i].ux;
    zdf_add_quant_part_file( &part_file, (char*)quants[1], data, spec ->np );

    for( i = 0; i < spec ->np; i++ ) data[i] = spec -> part[i].uy;
    zdf_add_quant_part_file( &part_file, (char*)quants[2], data, spec ->np );

    for( i = 0; i < spec ->np; i++ ) data[i] = spec -> part[i].uz;
    zdf_add_quant_part_file( &part_file, (char*)quants[3], data, spec ->np );

    free( data ); zdf_close_file( &part_file );
}

void spec_rep_charge( const t_species *spec )
{
    size_t size = ( spec -> nx + 1 ) * sizeof( float );
    float *charge = (float*) malloc( size );
    memset( charge, 0, size );
    spec_deposit_charge( spec, charge );
    float buffer[ spec -> nx ];
    for ( int i = 0; i < spec->nx; i++ ) buffer[i] = charge[i];
    free( charge );

    t_zdf_grid_axis axis;
    axis.min = spec -> n_move * spec -> dx; axis.max = spec->box + spec -> n_move * spec -> dx;
    axis.name = (char*)"x"; axis.label = (char*)"x"; axis.units = (char*)"c/\\omega_p";

    char name[128], label[128];
    snprintf(name, 128, "%s-charge", spec -> name);
    snprintf(label, 128, "%s \\rho", spec -> name);

    t_zdf_grid_info info;
    info.ndims = 1; info.name = name; info.label = label; info.units = (char*)"n_e";
    info.axis  = &axis; info.count[0] = spec->nx;

    t_zdf_iteration iter;
    iter.name = (char*)"ITERATION"; iter.n = spec->iter; iter.t = spec -> iter * spec -> dt; iter.time_units = (char*)"1/\\omega_p";

    char path[1024];
    snprintf(path, 1024, "CHARGE/%s", spec -> name );
    zdf_save_grid( (void *) buffer, zdf_float32, &info, &iter, path );
}

void spec_pha_axis( const t_species *spec, int i0, int np, int quant, float * restrict axis )
{
    switch (quant) {
        case X1: for (int i = 0; i < np; i++) axis[i] = ( spec -> part[i0+i].x + spec -> part[i0+i].ix ) * spec -> dx; break;
        case U1: for (int i = 0; i < np; i++) axis[i] = spec -> part[i0+i].ux; break;
        case U2: for (int i = 0; i < np; i++) axis[i] = spec -> part[i0+i].uy; break;
        case U3: for (int i = 0; i < np; i++) axis[i] = spec -> part[i0+i].uz; break;
    }
}

const char * spec_pha_axis_units( int quant ) {
    switch (quant) {
        case X1: return("c/\\omega_p");
        case U1: case U2: case U3: return("m_e c");
    }
    return("");
}

void spec_deposit_pha( const t_species *spec, const int rep_type, const int pha_nx[], const float pha_range[][2], float* restrict buf )
{
    const int BUF_SIZE = 1024;
    float pha_x1[BUF_SIZE], pha_x2[BUF_SIZE];
    const int nrow = pha_nx[0];
    const int quant1 = rep_type & 0x000F;
    const int quant2 = (rep_type & 0x00F0)>>4;
    const float x1min = pha_range[0][0]; const float x2min = pha_range[1][0];
    const float rdx1 = pha_nx[0] / ( pha_range[0][1] - pha_range[0][0] );
    const float rdx2 = pha_nx[1] / ( pha_range[1][1] - pha_range[1][0] );

    for ( int i = 0; i<spec->np; i+=BUF_SIZE ) {
        int np = ( i + BUF_SIZE > spec->np )? spec->np - i : BUF_SIZE;
        spec_pha_axis( spec, i, np, quant1, pha_x1 );
        spec_pha_axis( spec, i, np, quant2, pha_x2 );

        for ( int k = 0; k < np; k++ ) {
            float nx1 = ( pha_x1[k] - x1min ) * rdx1;
            float nx2 = ( pha_x2[k] - x2min ) * rdx2;
            int i1 = (int)(nx1 + 0.5f); int i2 = (int)(nx2 + 0.5f);
            float w1 = nx1 - i1 + 0.5f; float w2 = nx2 - i2 + 0.5f;
            int idx = i1 + nrow*i2;

            if ( i2 >= 0 && i2 < pha_nx[1] ) {
                if (i1 >= 0 && i1 < pha_nx[0]) buf[ idx ] += (1.0f-w1)*(1.0f-w2)*spec->q;
                if (i1+1 >= 0 && i1+1 < pha_nx[0] ) buf[ idx + 1 ] += w1*(1.0f-w2)*spec->q;
            }
            idx += nrow;
            if ( i2+1 >= 0 && i2+1 < pha_nx[1] ) {
                if (i1 >= 0 && i1 < pha_nx[0]) buf[ idx ] += (1.0f-w1)*w2*spec->q;
                if (i1+1 >= 0 && i1+1 < pha_nx[0] ) buf[ idx + 1 ] += w1*w2*spec->q;
            }
        }
    }
}

void spec_rep_pha( const t_species *spec, const int rep_type, const int pha_nx[], const float pha_range[][2] )
{
    float* restrict buf = (float*) malloc( pha_nx[0] * pha_nx[1] * sizeof( float ));
    memset( buf, 0, pha_nx[0] * pha_nx[1] * sizeof( float ));
    spec_deposit_pha( spec, rep_type, pha_nx, pha_range, buf );

    int quant1 = rep_type & 0x000F;
    int quant2 = (rep_type & 0x00F0)>>4;

    const char * pha_ax1_units = spec_pha_axis_units(quant1);
    const char * pha_ax2_units = spec_pha_axis_units(quant2);

    char const * const pha_ax_name[] = {"x1","x2","x3","u1","u2","u3"};
    char const * const pha_ax_label[] = {"x","y","z","u_x","u_y","u_z"};

    t_zdf_grid_axis axis[2];
    axis[0].min = pha_range[0][0]; axis[0].max = pha_range[0][1];
    axis[0].name = (char *) pha_ax_name[ quant1 - 1 ]; axis[0].label = (char *) pha_ax_label[ quant1 - 1 ]; axis[0].units = (char *) pha_ax1_units;

    axis[1].min = pha_range[1][0]; axis[1].max = pha_range[1][1];
    axis[1].name = (char *) pha_ax_name[ quant2 - 1 ]; axis[1].label = (char *) pha_ax_label[ quant2 - 1 ]; axis[1].units = (char *) pha_ax2_units;

    char pha_name[64], pha_label[64];
    snprintf( pha_name, 64,"%s-%s%s", spec -> name, pha_ax_name[quant1-1], pha_ax_name[quant2-1] );
    snprintf( pha_label, 64,"%s %s-%s", spec -> name, pha_ax_label[quant1-1], pha_ax_label[quant2-1] );

    t_zdf_grid_info info;
    info.ndims = 2; info.name = pha_name; info.label = pha_label; info.units = (char*)"a.u.";
    info.axis  = axis; info.count[0] = pha_nx[0]; info.count[1] = pha_nx[1];

    t_zdf_iteration iter;
    iter.name = (char*)"ITERATION"; iter.n = spec->iter; iter.t = spec -> iter * spec -> dt; iter.time_units = (char*)"1/\\omega_p";

    char path[1024];
    snprintf(path, 1024, "PHASESPACE/%s", spec -> name );
    zdf_save_grid( (void *) buf, zdf_float32, &info, &iter, path );
    free( buf );
}

void spec_report( const t_species *spec, const int rep_type, const int pha_nx[], const float pha_range[][2] )
{
    switch (rep_type & 0xF000) {
        case CHARGE: spec_rep_charge( spec ); break;
        case PHA: spec_rep_pha( spec, rep_type, pha_nx, pha_range ); break;
        case PARTICLES: spec_rep_particles( spec ); break;
    }
}
