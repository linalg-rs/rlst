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

/* > \brief \b CGBEQU */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CGBEQU + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cgbequ.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cgbequ.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cgbequ.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CGBEQU( M, N, KL, KU, AB, LDAB, R, C, ROWCND, COLCND, */
/*                          AMAX, INFO ) */

/*       INTEGER            INFO, KL, KU, LDAB, M, N */
/*       REAL               AMAX, COLCND, ROWCND */
/*       REAL               C( * ), R( * ) */
/*       COMPLEX            AB( LDAB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CGBEQU computes row and column scalings intended to equilibrate an */
/* > M-by-N band matrix A and reduce its condition number.  R returns the */
/* > row scale factors and C the column scale factors, chosen to try to */
/* > make the largest element in each row and column of the matrix B with */
/* > elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1. */
/* > */
/* > R(i) and C(j) are restricted to be between SMLNUM = smallest safe */
/* > number and BIGNUM = largest safe number.  Use of these scaling */
/* > factors is not guaranteed to reduce the condition number of A but */
/* > works well in practice. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] KL */
/* > \verbatim */
/* >          KL is INTEGER */
/* >          The number of subdiagonals within the band of A.  KL >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] KU */
/* > \verbatim */
/* >          KU is INTEGER */
/* >          The number of superdiagonals within the band of A.  KU >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] AB */
/* > \verbatim */
/* >          AB is COMPLEX array, dimension (LDAB,N) */
/* >          The band matrix A, stored in rows 1 to KL+KU+1.  The j-th */
/* >          column of A is stored in the j-th column of the array AB as */
/* >          follows: */
/* >          AB(ku+1+i-j,j) = A(i,j) for f2cmax(1,j-ku)<=i<=f2cmin(m,j+kl). */
/* > \endverbatim */
/* > */
/* > \param[in] LDAB */
/* > \verbatim */
/* >          LDAB is INTEGER */
/* >          The leading dimension of the array AB.  LDAB >= KL+KU+1. */
/* > \endverbatim */
/* > */
/* > \param[out] R */
/* > \verbatim */
/* >          R is REAL array, dimension (M) */
/* >          If INFO = 0, or INFO > M, R contains the row scale factors */
/* >          for A. */
/* > \endverbatim */
/* > */
/* > \param[out] C */
/* > \verbatim */
/* >          C is REAL array, dimension (N) */
/* >          If INFO = 0, C contains the column scale factors for A. */
/* > \endverbatim */
/* > */
/* > \param[out] ROWCND */
/* > \verbatim */
/* >          ROWCND is REAL */
/* >          If INFO = 0 or INFO > M, ROWCND contains the ratio of the */
/* >          smallest R(i) to the largest R(i).  If ROWCND >= 0.1 and */
/* >          AMAX is neither too large nor too small, it is not worth */
/* >          scaling by R. */
/* > \endverbatim */
/* > */
/* > \param[out] COLCND */
/* > \verbatim */
/* >          COLCND is REAL */
/* >          If INFO = 0, COLCND contains the ratio of the smallest */
/* >          C(i) to the largest C(i).  If COLCND >= 0.1, it is not */
/* >          worth scaling by C. */
/* > \endverbatim */
/* > */
/* > \param[out] AMAX */
/* > \verbatim */
/* >          AMAX is REAL */
/* >          Absolute value of largest matrix element.  If AMAX is very */
/* >          close to overflow or very close to underflow, the matrix */
/* >          should be scaled. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, and i is */
/* >                <= M:  the i-th row of A is exactly zero */
/* >                >  M:  the (i-M)-th column of A is exactly zero */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexGBcomputational */

/*  ===================================================================== */
void  cgbequ_(integer *m, integer *n, integer *kl, integer *ku,
	 complex *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real 
	*colcnd, real *amax, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2, r__3, r__4;

    /* Local variables */
    integer i__, j, kd;
    real rcmin, rcmax;
    extern real slamch_(char *);
    extern void  xerbla_(char *, integer *);
    real bignum, smlnum;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --r__;
    --c__;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*kl < 0) {
	*info = -3;
    } else if (*ku < 0) {
	*info = -4;
    } else if (*ldab < *kl + *ku + 1) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGBEQU", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	*rowcnd = 1.f;
	*colcnd = 1.f;
	*amax = 0.f;
	return;
    }

/*     Get machine constants. */

    smlnum = slamch_("S");
    bignum = 1.f / smlnum;

/*     Compute row scale factors. */

    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__[i__] = 0.f;
/* L10: */
    }

/*     Find the maximum element in each row. */

    kd = *ku + 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	i__2 = j - *ku;
/* Computing MIN */
	i__4 = j + *kl;
	i__3 = f2cmin(i__4,*m);
	for (i__ = f2cmax(i__2,1); i__ <= i__3; ++i__) {
/* Computing MAX */
	    i__2 = kd + i__ - j + j * ab_dim1;
	    r__3 = r__[i__], r__4 = (r__1 = ab[i__2].r, abs(r__1)) + (r__2 = 
		    r_imag(&ab[kd + i__ - j + j * ab_dim1]), abs(r__2));
	    r__[i__] = f2cmax(r__3,r__4);
/* L20: */
	}
/* L30: */
    }

/*     Find the maximum and minimum scale factors. */

    rcmin = bignum;
    rcmax = 0.f;
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	r__1 = rcmax, r__2 = r__[i__];
	rcmax = f2cmax(r__1,r__2);
/* Computing MIN */
	r__1 = rcmin, r__2 = r__[i__];
	rcmin = f2cmin(r__1,r__2);
/* L40: */
    }
    *amax = rcmax;

    if (rcmin == 0.f) {

/*        Find the first zero scale factor and return an error code. */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (r__[i__] == 0.f) {
		*info = i__;
		return;
	    }
/* L50: */
	}
    } else {

/*        Invert the scale factors. */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MIN */
/* Computing MAX */
	    r__2 = r__[i__];
	    r__1 = f2cmax(r__2,smlnum);
	    r__[i__] = 1.f / f2cmin(r__1,bignum);
/* L60: */
	}

/*        Compute ROWCND = f2cmin(R(I)) / f2cmax(R(I)) */

	*rowcnd = f2cmax(rcmin,smlnum) / f2cmin(rcmax,bignum);
    }

/*     Compute column scale factors */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	c__[j] = 0.f;
/* L70: */
    }

/*     Find the maximum element in each column, */
/*     assuming the row scaling computed above. */

    kd = *ku + 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	i__3 = j - *ku;
/* Computing MIN */
	i__4 = j + *kl;
	i__2 = f2cmin(i__4,*m);
	for (i__ = f2cmax(i__3,1); i__ <= i__2; ++i__) {
/* Computing MAX */
	    i__3 = kd + i__ - j + j * ab_dim1;
	    r__3 = c__[j], r__4 = ((r__1 = ab[i__3].r, abs(r__1)) + (r__2 = 
		    r_imag(&ab[kd + i__ - j + j * ab_dim1]), abs(r__2))) * 
		    r__[i__];
	    c__[j] = f2cmax(r__3,r__4);
/* L80: */
	}
/* L90: */
    }

/*     Find the maximum and minimum scale factors. */

    rcmin = bignum;
    rcmax = 0.f;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	r__1 = rcmin, r__2 = c__[j];
	rcmin = f2cmin(r__1,r__2);
/* Computing MAX */
	r__1 = rcmax, r__2 = c__[j];
	rcmax = f2cmax(r__1,r__2);
/* L100: */
    }

    if (rcmin == 0.f) {

/*        Find the first zero scale factor and return an error code. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (c__[j] == 0.f) {
		*info = *m + j;
		return;
	    }
/* L110: */
	}
    } else {

/*        Invert the scale factors. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
/* Computing MAX */
	    r__2 = c__[j];
	    r__1 = f2cmax(r__2,smlnum);
	    c__[j] = 1.f / f2cmin(r__1,bignum);
/* L120: */
	}

/*        Compute COLCND = f2cmin(C(J)) / f2cmax(C(J)) */

	*colcnd = f2cmax(rcmin,smlnum) / f2cmin(rcmax,bignum);
    }

    return;

/*     End of CGBEQU */

} /* cgbequ_ */

