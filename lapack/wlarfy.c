#define len_trim__(cad,len) ({                           integer _r=0,i;                           for(i=0; i<(len) && (cad)[i]; i++)                             if((cad)[i] != ' ') _r=i;                           _r+1; })
#define ceiling_(a) (myceil(*(a)))
#define myceil(a) (sizeof(a) == sizeof(float) ? ceilf(a) : ceil(a))
#include <math.h>
/*  -- translated by f2c (version 20200916).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_QUAD
#include "f2c.h"

/* Table of constant values */

static quadcomplex c_b1 = {1.,0.};
static quadcomplex c_b2 = {0.,0.};
static integer c__1 = 1;

/* > \brief \b ZLARFY */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLARFY( UPLO, N, V, INCV, TAU, C, LDC, WORK ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INCV, LDC, N */
/*       COMPLEX*16         TAU */
/*       COMPLEX*16         C( LDC, * ), V( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLARFY applies an elementary reflector, or Householder matrix, H, */
/* > to an n x n Hermitian matrix C, from both the left and the right. */
/* > */
/* > H is represented in the form */
/* > */
/* >    H = I - tau * v * v' */
/* > */
/* > where  tau  is a scalar and  v  is a vector. */
/* > */
/* > If  tau  is  zero, then  H  is taken to be the unit matrix. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the upper or lower triangular part of the */
/* >          Hermitian matrix C is stored. */
/* >          = 'U':  Upper triangle */
/* >          = 'L':  Lower triangle */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of rows and columns of the matrix C.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] V */
/* > \verbatim */
/* >          V is COMPLEX*16 array, dimension */
/* >                  (1 + (N-1)*abs(INCV)) */
/* >          The vector v as described above. */
/* > \endverbatim */
/* > */
/* > \param[in] INCV */
/* > \verbatim */
/* >          INCV is INTEGER */
/* >          The increment between successive elements of v.  INCV must */
/* >          not be zero. */
/* > \endverbatim */
/* > */
/* > \param[in] TAU */
/* > \verbatim */
/* >          TAU is COMPLEX*16 */
/* >          The value tau as described above. */
/* > \endverbatim */
/* > */
/* > \param[in,out] C */
/* > \verbatim */
/* >          C is COMPLEX*16 array, dimension (LDC, N) */
/* >          On entry, the matrix C. */
/* >          On exit, C is overwritten by H * C * H'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDC */
/* > \verbatim */
/* >          LDC is INTEGER */
/* >          The leading dimension of the array C.  LDC >= f2cmax( 1, N ). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (N) */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16_eig */

/*  ===================================================================== */
void  wlarfy_(char *uplo, integer *n, quadcomplex *v, 
	integer *incv, quadcomplex *tau, quadcomplex *c__, integer *ldc, 
	quadcomplex *work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    quadcomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    extern void  wher2_(char *, integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *);
    quadcomplex alpha;
    extern /* Double Complex */ VOID wqotc_(quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *);
    extern void  whemv_(char *, integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, quadcomplex *, integer *), waxpy_(
	    integer *, quadcomplex *, quadcomplex *, integer *, 
	    quadcomplex *, integer *);


/*  -- LAPACK test routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    if (tau->r == 0. && tau->i == 0.) {
	return;
    }

/*     Form  w:= C * v */

    whemv_(uplo, n, &c_b1, &c__[c_offset], ldc, &v[1], incv, &c_b2, &work[1], 
	    &c__1);

    z__3.r = -.5, z__3.i = -0.;
    z__2.r = z__3.r * tau->r - z__3.i * tau->i, z__2.i = z__3.r * tau->i + 
	    z__3.i * tau->r;
    wqotc_(&z__4, n, &work[1], &c__1, &v[1], incv);
    z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r * z__4.i + 
	    z__2.i * z__4.r;
    alpha.r = z__1.r, alpha.i = z__1.i;
    waxpy_(n, &alpha, &v[1], incv, &work[1], &c__1);

/*     C := C - v * w' - w * v' */

    z__1.r = -tau->r, z__1.i = -tau->i;
    wher2_(uplo, n, &z__1, &v[1], incv, &work[1], &c__1, &c__[c_offset], ldc);

    return;

/*     End of ZLARFY */

} /* wlarfy_ */

