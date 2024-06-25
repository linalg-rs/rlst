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

/* > \brief \b CSYR performs the symmetric rank-1 update of a complex symmetric matrix. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CSYR + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/csyr.f"
> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/csyr.f"
> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/csyr.f"
> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CSYR( UPLO, N, ALPHA, X, INCX, A, LDA ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INCX, LDA, N */
/*       COMPLEX            ALPHA */
/*       COMPLEX            A( LDA, * ), X( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CSYR   performs the symmetric rank 1 operation */
/* > */
/* >    A := alpha*x*x**H + A, */
/* > */
/* > where alpha is a complex scalar, x is an n element vector and A is an */
/* > n by n symmetric matrix. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >           On entry, UPLO specifies whether the upper or lower */
/* >           triangular part of the array A is to be referenced as */
/* >           follows: */
/* > */
/* >              UPLO = 'U' or 'u'   Only the upper triangular part of A */
/* >                                  is to be referenced. */
/* > */
/* >              UPLO = 'L' or 'l'   Only the lower triangular part of A */
/* >                                  is to be referenced. */
/* > */
/* >           Unchanged on exit. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >           On entry, N specifies the order of the matrix A. */
/* >           N must be at least zero. */
/* >           Unchanged on exit. */
/* > \endverbatim */
/* > */
/* > \param[in] ALPHA */
/* > \verbatim */
/* >          ALPHA is COMPLEX */
/* >           On entry, ALPHA specifies the scalar alpha. */
/* >           Unchanged on exit. */
/* > \endverbatim */
/* > */
/* > \param[in] X */
/* > \verbatim */
/* >          X is COMPLEX array, dimension at least */
/* >           ( 1 + ( N - 1 )*abs( INCX ) ). */
/* >           Before entry, the incremented array X must contain the N- */
/* >           element vector x. */
/* >           Unchanged on exit. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >           On entry, INCX specifies the increment for the elements of */
/* >           X. INCX must not be zero. */
/* >           Unchanged on exit. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension ( LDA, N ) */
/* >           Before entry, with  UPLO = 'U' or 'u', the leading n by n */
/* >           upper triangular part of the array A must contain the upper */
/* >           triangular part of the symmetric matrix and the strictly */
/* >           lower triangular part of A is not referenced. On exit, the */
/* >           upper triangular part of the array A is overwritten by the */
/* >           upper triangular part of the updated matrix. */
/* >           Before entry, with UPLO = 'L' or 'l', the leading n by n */
/* >           lower triangular part of the array A must contain the lower */
/* >           triangular part of the symmetric matrix and the strictly */
/* >           upper triangular part of A is not referenced. On exit, the */
/* >           lower triangular part of the array A is overwritten by the */
/* >           lower triangular part of the updated matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >           On entry, LDA specifies the first dimension of A as declared */
/* >           in the calling (sub) program. LDA must be at least */
/* >           f2cmax( 1, N ). */
/* >           Unchanged on exit. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexSYauxiliary */

/*  ===================================================================== */
void  csyr_(char *uplo, integer *n, complex *alpha, complex *x,
	 integer *incx, complex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1, q__2;

    /* Local variables */
    integer i__, j, ix, jx, kx, info;
    complex temp;
    extern logical lsame_(char *, char *);
    extern void  xerbla_(char *, integer *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/* ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --x;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*lda < f2cmax(1,*n)) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("CSYR  ", &info);
	return;
    }

/*     Quick return if possible. */

    if (*n == 0 || alpha->r == 0.f && alpha->i == 0.f) {
	return;
    }

/*     Set the start point in X if the increment is not unity. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the triangular part */
/*     of A. */

    if (lsame_(uplo, "U")) {

/*        Form  A  when A is stored in upper triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = j;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = i__;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L10: */
		    }
		}
/* L20: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = jx;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    ix = kx;
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = ix;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			ix += *incx;
/* L30: */
		    }
		}
		jx += *incx;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when A is stored in lower triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = j;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = i__;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L50: */
		    }
		}
/* L60: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = jx;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    ix = jx;
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = ix;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			ix += *incx;
/* L70: */
		    }
		}
		jx += *incx;
/* L80: */
	    }
	}
    }

    return;

/*     End of CSYR */

} /* csyr_ */

