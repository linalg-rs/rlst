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

#define __LAPACK_PRECISION_DOUBLE
#include "f2c.h"

/* > \brief \b DGBEQUB */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DGBEQUB + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgbequb
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgbequb
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgbequb
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DGBEQUB( M, N, KL, KU, AB, LDAB, R, C, ROWCND, COLCND, */
/*                           AMAX, INFO ) */

/*       INTEGER            INFO, KL, KU, LDAB, M, N */
/*       DOUBLE PRECISION   AMAX, COLCND, ROWCND */
/*       DOUBLE PRECISION   AB( LDAB, * ), C( * ), R( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGBEQUB computes row and column scalings intended to equilibrate an */
/* > M-by-N matrix A and reduce its condition number.  R returns the row */
/* > scale factors and C the column scale factors, chosen to try to make */
/* > the largest element in each row and column of the matrix B with */
/* > elements B(i,j)=R(i)*A(i,j)*C(j) have an absolute value of at most */
/* > the radix. */
/* > */
/* > R(i) and C(j) are restricted to be a power of the radix between */
/* > SMLNUM = smallest safe number and BIGNUM = largest safe number.  Use */
/* > of these scaling factors is not guaranteed to reduce the condition */
/* > number of A but works well in practice. */
/* > */
/* > This routine differs from DGEEQU by restricting the scaling factors */
/* > to a power of the radix.  Barring over- and underflow, scaling by */
/* > these factors introduces no additional rounding errors.  However, the */
/* > scaled entries' magnitudes are no longer approximately 1 but lie */
/* > between M(sqrt)(radix) and 1/M(sqrt)(radix). */
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
/* >          AB is DOUBLE PRECISION array, dimension (LDAB,N) */
/* >          On entry, the matrix A in band storage, in rows 1 to KL+KU+1. */
/* >          The j-th column of A is stored in the j-th column of the */
/* >          array AB as follows: */
/* >          AB(KU+1+i-j,j) = A(i,j) for f2cmax(1,j-KU)<=i<=f2cmin(N,j+kl) */
/* > \endverbatim */
/* > */
/* > \param[in] LDAB */
/* > \verbatim */
/* >          LDAB is INTEGER */
/* >          The leading dimension of the array A.  LDAB >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] R */
/* > \verbatim */
/* >          R is DOUBLE PRECISION array, dimension (M) */
/* >          If INFO = 0 or INFO > M, R contains the row scale factors */
/* >          for A. */
/* > \endverbatim */
/* > */
/* > \param[out] C */
/* > \verbatim */
/* >          C is DOUBLE PRECISION array, dimension (N) */
/* >          If INFO = 0,  C contains the column scale factors for A. */
/* > \endverbatim */
/* > */
/* > \param[out] ROWCND */
/* > \verbatim */
/* >          ROWCND is DOUBLE PRECISION */
/* >          If INFO = 0 or INFO > M, ROWCND contains the ratio of the */
/* >          smallest R(i) to the largest R(i).  If ROWCND >= 0.1 and */
/* >          AMAX is neither too large nor too small, it is not worth */
/* >          scaling by R. */
/* > \endverbatim */
/* > */
/* > \param[out] COLCND */
/* > \verbatim */
/* >          COLCND is DOUBLE PRECISION */
/* >          If INFO = 0, COLCND contains the ratio of the smallest */
/* >          C(i) to the largest C(i).  If COLCND >= 0.1, it is not */
/* >          worth scaling by C. */
/* > \endverbatim */
/* > */
/* > \param[out] AMAX */
/* > \verbatim */
/* >          AMAX is DOUBLE PRECISION */
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
/* >          > 0:  if INFO = i,  and i is */
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

/* > \ingroup doubleGBcomputational */

/*  ===================================================================== */
void  dgbequb_(integer *m, integer *n, integer *kl, integer *
	ku, doublereal *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, integer *
	info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3, i__4;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    integer i__, j, kd;
    doublereal radix, rcmin, rcmax;
    extern doublereal dlamch_(char *);
    extern void  xerbla_(char *, integer *);
    doublereal bignum, logrdx, smlnum;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

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
	xerbla_("DGBEQUB", &i__1);
	return;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0) {
	*rowcnd = 1.;
	*colcnd = 1.;
	*amax = 0.;
	return;
    }

/*     Get machine constants.  Assume SMLNUM is a power of the radix. */

    smlnum = dlamch_("S");
    bignum = 1. / smlnum;
    radix = dlamch_("B");
    logrdx = M(log)(radix);

/*     Compute row scale factors. */

    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__[i__] = 0.;
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
	    d__2 = r__[i__], d__3 = (d__1 = ab[kd + i__ - j + j * ab_dim1], 
		    abs(d__1));
	    r__[i__] = f2cmax(d__2,d__3);
/* L20: */
	}
/* L30: */
    }
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (r__[i__] > 0.) {
	    i__3 = (integer) (M(log)(r__[i__]) / logrdx);
	    r__[i__] = pow_di(&radix, &i__3);
	}
    }

/*     Find the maximum and minimum scale factors. */

    rcmin = bignum;
    rcmax = 0.;
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	d__1 = rcmax, d__2 = r__[i__];
	rcmax = f2cmax(d__1,d__2);
/* Computing MIN */
	d__1 = rcmin, d__2 = r__[i__];
	rcmin = f2cmin(d__1,d__2);
/* L40: */
    }
    *amax = rcmax;

    if (rcmin == 0.) {

/*        Find the first zero scale factor and return an error code. */

	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (r__[i__] == 0.) {
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
	    d__2 = r__[i__];
	    d__1 = f2cmax(d__2,smlnum);
	    r__[i__] = 1. / f2cmin(d__1,bignum);
/* L60: */
	}

/*        Compute ROWCND = f2cmin(R(I)) / f2cmax(R(I)). */

	*rowcnd = f2cmax(rcmin,smlnum) / f2cmin(rcmax,bignum);
    }

/*     Compute column scale factors. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	c__[j] = 0.;
/* L70: */
    }

/*     Find the maximum element in each column, */
/*     assuming the row scaling computed above. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	i__3 = j - *ku;
/* Computing MIN */
	i__4 = j + *kl;
	i__2 = f2cmin(i__4,*m);
	for (i__ = f2cmax(i__3,1); i__ <= i__2; ++i__) {
/* Computing MAX */
	    d__2 = c__[j], d__3 = (d__1 = ab[kd + i__ - j + j * ab_dim1], abs(
		    d__1)) * r__[i__];
	    c__[j] = f2cmax(d__2,d__3);
/* L80: */
	}
	if (c__[j] > 0.) {
	    i__2 = (integer) (M(log)(c__[j]) / logrdx);
	    c__[j] = pow_di(&radix, &i__2);
	}
/* L90: */
    }

/*     Find the maximum and minimum scale factors. */

    rcmin = bignum;
    rcmax = 0.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
	d__1 = rcmin, d__2 = c__[j];
	rcmin = f2cmin(d__1,d__2);
/* Computing MAX */
	d__1 = rcmax, d__2 = c__[j];
	rcmax = f2cmax(d__1,d__2);
/* L100: */
    }

    if (rcmin == 0.) {

/*        Find the first zero scale factor and return an error code. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (c__[j] == 0.) {
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
	    d__2 = c__[j];
	    d__1 = f2cmax(d__2,smlnum);
	    c__[j] = 1. / f2cmin(d__1,bignum);
/* L120: */
	}

/*        Compute COLCND = f2cmin(C(J)) / f2cmax(C(J)). */

	*colcnd = f2cmax(rcmin,smlnum) / f2cmin(rcmax,bignum);
    }

    return;

/*     End of DGBEQUB */

} /* dgbequb_ */
