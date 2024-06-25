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

#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"

/* Table of constant values */

static complex c_b1 = {1.f,0.f};
static complex c_b2 = {0.f,0.f};
static integer c__1 = 1;

/* > \brief \b CLARFY */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CLARFY( UPLO, N, V, INCV, TAU, C, LDC, WORK ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INCV, LDC, N */
/*       COMPLEX            TAU */
/*       COMPLEX            C( LDC, * ), V( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CLARFY applies an elementary reflector, or Householder matrix, H, */
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
/* >          V is COMPLEX array, dimension */
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
/* >          TAU is COMPLEX */
/* >          The value tau as described above. */
/* > \endverbatim */
/* > */
/* > \param[in,out] C */
/* > \verbatim */
/* >          C is COMPLEX array, dimension (LDC, N) */
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
/* >          WORK is COMPLEX array, dimension (N) */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex_eig */

/*  ===================================================================== */
void  clarfy_(char *uplo, integer *n, complex *v, integer *
	incv, complex *tau, complex *c__, integer *ldc, complex *work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    complex q__1, q__2, q__3, q__4;

    /* Local variables */
    extern void  cher2_(char *, integer *, complex *, complex *
	    , integer *, complex *, integer *, complex *, integer *);
    complex alpha;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern void  chemv_(char *, integer *, complex *, complex *
	    , integer *, complex *, integer *, complex *, complex *, integer *
	    ), caxpy_(integer *, complex *, complex *, integer *, 
	    complex *, integer *);


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
    if (tau->r == 0.f && tau->i == 0.f) {
	return;
    }

/*     Form  w:= C * v */

    chemv_(uplo, n, &c_b1, &c__[c_offset], ldc, &v[1], incv, &c_b2, &work[1], 
	    &c__1);

    q__3.r = -.5f, q__3.i = -0.f;
    q__2.r = q__3.r * tau->r - q__3.i * tau->i, q__2.i = q__3.r * tau->i + 
	    q__3.i * tau->r;
    cdotc_(&q__4, n, &work[1], &c__1, &v[1], incv);
    q__1.r = q__2.r * q__4.r - q__2.i * q__4.i, q__1.i = q__2.r * q__4.i + 
	    q__2.i * q__4.r;
    alpha.r = q__1.r, alpha.i = q__1.i;
    caxpy_(n, &alpha, &v[1], incv, &work[1], &c__1);

/*     C := C - v * w' - w * v' */

    q__1.r = -tau->r, q__1.i = -tau->i;
    cher2_(uplo, n, &q__1, &v[1], incv, &work[1], &c__1, &c__[c_offset], ldc);

    return;

/*     End of CLARFY */

} /* clarfy_ */

