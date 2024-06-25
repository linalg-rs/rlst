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

static quadreal c_b2 = 1.;
static quadreal c_b3 = 0.;
static integer c__1 = 1;

/* > \brief \b DLARFY */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLARFY( UPLO, N, V, INCV, TAU, C, LDC, WORK ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INCV, LDC, N */
/*       DOUBLE PRECISION   TAU */
/*       DOUBLE PRECISION   C( LDC, * ), V( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLARFY applies an elementary reflector, or Householder matrix, H, */
/* > to an n x n symmetric matrix C, from both the left and the right. */
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
/* >          symmetric matrix C is stored. */
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
/* >          V is DOUBLE PRECISION array, dimension */
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
/* >          TAU is DOUBLE PRECISION */
/* >          The value tau as described above. */
/* > \endverbatim */
/* > */
/* > \param[in,out] C */
/* > \verbatim */
/* >          C is DOUBLE PRECISION array, dimension (LDC, N) */
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
/* >          WORK is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup double_eig */

/*  ===================================================================== */
void  qlarfy_(char *uplo, integer *n, quadreal *v, integer *
	incv, quadreal *tau, quadreal *c__, integer *ldc, quadreal *
	work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    quadreal d__1;

    /* Local variables */
    extern quadreal qdot_(integer *, quadreal *, integer *, quadreal *, 
	    integer *);
    extern void  qsyr2_(char *, integer *, quadreal *, 
	    quadreal *, integer *, quadreal *, integer *, quadreal *, 
	    integer *);
    quadreal alpha;
    extern void  qaxpy_(integer *, quadreal *, quadreal *, 
	    integer *, quadreal *, integer *), qsymv_(char *, integer *, 
	    quadreal *, quadreal *, integer *, quadreal *, integer *, 
	    quadreal *, quadreal *, integer *);


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
    if (*tau == 0.) {
	return;
    }

/*     Form  w:= C * v */

    qsymv_(uplo, n, &c_b2, &c__[c_offset], ldc, &v[1], incv, &c_b3, &work[1], 
	    &c__1);

    alpha = *tau * -.5 * qdot_(n, &work[1], &c__1, &v[1], incv);
    qaxpy_(n, &alpha, &v[1], incv, &work[1], &c__1);

/*     C := C - v * w' - w * v' */

    d__1 = -(*tau);
    qsyr2_(uplo, n, &d__1, &v[1], incv, &work[1], &c__1, &c__[c_offset], ldc);

    return;

/*     End of DLARFY */

} /* qlarfy_ */

