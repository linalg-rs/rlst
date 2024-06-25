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

static integer c__1 = 1;

/* > \brief \b SSYEQUB */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SSYEQUB + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ssyequb
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ssyequb
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ssyequb
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SSYEQUB( UPLO, N, A, LDA, S, SCOND, AMAX, WORK, INFO ) */

/*       INTEGER            INFO, LDA, N */
/*       REAL               AMAX, SCOND */
/*       CHARACTER          UPLO */
/*       REAL               A( LDA, * ), S( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SSYEQUB computes row and column scalings intended to equilibrate a */
/* > symmetric matrix A (with respect to the Euclidean norm) and reduce */
/* > its condition number. The scale factors S are computed by the BIN */
/* > algorithm (see references) so that the scaled matrix B with elements */
/* > B(i,j) = S(i)*A(i,j)*S(j) has a condition number within a factor N of */
/* > the smallest possible condition number over all possible diagonal */
/* > scalings. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  Upper triangle of A is stored; */
/* >          = 'L':  Lower triangle of A is stored. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is REAL array, dimension (LDA,N) */
/* >          The N-by-N symmetric matrix whose scaling factors are to be */
/* >          computed. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] S */
/* > \verbatim */
/* >          S is REAL array, dimension (N) */
/* >          If INFO = 0, S contains the scale factors for A. */
/* > \endverbatim */
/* > */
/* > \param[out] SCOND */
/* > \verbatim */
/* >          SCOND is REAL */
/* >          If INFO = 0, S contains the ratio of the smallest S(i) to */
/* >          the largest S(i). If SCOND >= 0.1 and AMAX is neither too */
/* >          large nor too small, it is not worth scaling by S. */
/* > \endverbatim */
/* > */
/* > \param[out] AMAX */
/* > \verbatim */
/* >          AMAX is REAL */
/* >          Largest absolute value of any matrix element. If AMAX is */
/* >          very close to overflow or very close to underflow, the */
/* >          matrix should be scaled. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is REAL array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, the i-th diagonal element is nonpositive. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2017 */

/* > \ingroup realSYcomputational */

/* > \par References: */
/*  ================ */
/* > */
/* >  Livne, O.E. and Golub, G.H., "Scaling by Binormalization", \n */
/* >  Numerical Algorithms, vol. 35, no. 1, pp. 97-120, January 2004. \n */
/* >  DOI 10.1023/B:NUMA.0000016606.32820.69 \n */
/* >  Tech report version: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.3.1679 */
/* > */
/*  ===================================================================== */
void  ssyequb_(char *uplo, integer *n, real *a, integer *lda, 
	real *s, real *scond, real *amax, real *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1, r__2, r__3;

    /* Local variables */
    real d__;
    integer i__, j;
    real t, u, c0, c1, c2, si;
    logical up;
    real avg, std, tol, base;
    integer iter;
    real smin, smax, scale;
    extern logical lsame_(char *, char *);
    real sumsq;
    extern real slamch_(char *);
    extern void  xerbla_(char *, integer *);
    real bignum;
    extern void  slassq_(integer *, real *, integer *, real *, 
	    real *);
    real smlnum;


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --s;
    --work;

    /* Function Body */
    *info = 0;
    if (! (lsame_(uplo, "U") || lsame_(uplo, "L"))) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYEQUB", &i__1);
	return;
    }
    up = lsame_(uplo, "U");
    *amax = 0.f;

/*     Quick return if possible. */

    if (*n == 0) {
	*scond = 1.f;
	return;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	s[i__] = 0.f;
    }
    *amax = 0.f;
    if (up) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		r__2 = s[i__], r__3 = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		s[i__] = f2cmax(r__2,r__3);
/* Computing MAX */
		r__2 = s[j], r__3 = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		s[j] = f2cmax(r__2,r__3);
/* Computing MAX */
		r__2 = *amax, r__3 = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		*amax = f2cmax(r__2,r__3);
	    }
/* Computing MAX */
	    r__2 = s[j], r__3 = (r__1 = a[j + j * a_dim1], abs(r__1));
	    s[j] = f2cmax(r__2,r__3);
/* Computing MAX */
	    r__2 = *amax, r__3 = (r__1 = a[j + j * a_dim1], abs(r__1));
	    *amax = f2cmax(r__2,r__3);
	}
    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	    r__2 = s[j], r__3 = (r__1 = a[j + j * a_dim1], abs(r__1));
	    s[j] = f2cmax(r__2,r__3);
/* Computing MAX */
	    r__2 = *amax, r__3 = (r__1 = a[j + j * a_dim1], abs(r__1));
	    *amax = f2cmax(r__2,r__3);
	    i__2 = *n;
	    for (i__ = j + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		r__2 = s[i__], r__3 = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		s[i__] = f2cmax(r__2,r__3);
/* Computing MAX */
		r__2 = s[j], r__3 = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		s[j] = f2cmax(r__2,r__3);
/* Computing MAX */
		r__2 = *amax, r__3 = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		*amax = f2cmax(r__2,r__3);
	    }
	}
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	s[j] = 1.f / s[j];
    }
    tol = 1.f / M(sqrt)(*n * 2.f);
    for (iter = 1; iter <= 100; ++iter) {
	scale = 0.f;
	sumsq = 0.f;
/*        beta = |A|s */
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.f;
	}
	if (up) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    work[i__] += (r__1 = a[i__ + j * a_dim1], abs(r__1)) * s[
			    j];
		    work[j] += (r__1 = a[i__ + j * a_dim1], abs(r__1)) * s[
			    i__];
		}
		work[j] += (r__1 = a[j + j * a_dim1], abs(r__1)) * s[j];
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] += (r__1 = a[j + j * a_dim1], abs(r__1)) * s[j];
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    work[i__] += (r__1 = a[i__ + j * a_dim1], abs(r__1)) * s[
			    j];
		    work[j] += (r__1 = a[i__ + j * a_dim1], abs(r__1)) * s[
			    i__];
		}
	    }
	}
/*        avg = s^T beta / n */
	avg = 0.f;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    avg += s[i__] * work[i__];
	}
	avg /= *n;
	std = 0.f;
	i__1 = *n << 1;
	for (i__ = *n + 1; i__ <= i__1; ++i__) {
	    work[i__] = s[i__ - *n] * work[i__ - *n] - avg;
	}
	slassq_(n, &work[*n + 1], &c__1, &scale, &sumsq);
	std = scale * M(sqrt)(sumsq / *n);
	if (std < tol * avg) {
	    goto L999;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    t = (r__1 = a[i__ + i__ * a_dim1], abs(r__1));
	    si = s[i__];
	    c2 = (*n - 1) * t;
	    c1 = (*n - 2) * (work[i__] - t * si);
	    c0 = -(t * si) * si + work[i__] * 2 * si - *n * avg;
	    d__ = c1 * c1 - c0 * 4 * c2;
	    if (d__ <= 0.f) {
		*info = -1;
		return;
	    }
	    si = c0 * -2 / (c1 + M(sqrt)(d__));
	    d__ = si - s[i__];
	    u = 0.f;
	    if (up) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    t = (r__1 = a[j + i__ * a_dim1], abs(r__1));
		    u += s[j] * t;
		    work[j] += d__ * t;
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    t = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		    u += s[j] * t;
		    work[j] += d__ * t;
		}
	    } else {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    t = (r__1 = a[i__ + j * a_dim1], abs(r__1));
		    u += s[j] * t;
		    work[j] += d__ * t;
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    t = (r__1 = a[j + i__ * a_dim1], abs(r__1));
		    u += s[j] * t;
		    work[j] += d__ * t;
		}
	    }
	    avg += (u + work[i__]) * d__ / *n;
	    s[i__] = si;
	}
    }
L999:
    smlnum = slamch_("SAFEMIN");
    bignum = 1.f / smlnum;
    smin = bignum;
    smax = 0.f;
    t = 1.f / M(sqrt)(avg);
    base = slamch_("B");
    u = 1.f / M(log)(base);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = (integer) (u * M(log)(s[i__] * t));
	s[i__] = pow_ri(&base, &i__2);
/* Computing MIN */
	r__1 = smin, r__2 = s[i__];
	smin = f2cmin(r__1,r__2);
/* Computing MAX */
	r__1 = smax, r__2 = s[i__];
	smax = f2cmax(r__1,r__2);
    }
    *scond = f2cmax(smin,smlnum) / f2cmin(smax,bignum);

    return;
} /* ssyequb_ */
