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

/* > \brief \b DPOEQU */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DPOEQU + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dpoequ.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dpoequ.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dpoequ.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DPOEQU( N, A, LDA, S, SCOND, AMAX, INFO ) */

/*       INTEGER            INFO, LDA, N */
/*       DOUBLE PRECISION   AMAX, SCOND */
/*       DOUBLE PRECISION   A( LDA, * ), S( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DPOEQU computes row and column scalings intended to equilibrate a */
/* > symmetric positive definite matrix A and reduce its condition number */
/* > (with respect to the two-norm).  S contains the scale factors, */
/* > S(i) = 1/M(sqrt)(A(i,i)), chosen so that the scaled matrix B with */
/* > elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This */
/* > choice of S puts the condition number of B within a factor N of the */
/* > smallest possible condition number over all possible diagonal */
/* > scalings. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA,N) */
/* >          The N-by-N symmetric positive definite matrix whose scaling */
/* >          factors are to be computed.  Only the diagonal elements of A */
/* >          are referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] S */
/* > \verbatim */
/* >          S is DOUBLE PRECISION array, dimension (N) */
/* >          If INFO = 0, S contains the scale factors for A. */
/* > \endverbatim */
/* > */
/* > \param[out] SCOND */
/* > \verbatim */
/* >          SCOND is DOUBLE PRECISION */
/* >          If INFO = 0, S contains the ratio of the smallest S(i) to */
/* >          the largest S(i).  If SCOND >= 0.1 and AMAX is neither too */
/* >          large nor too small, it is not worth scaling by S. */
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
/* >          > 0:  if INFO = i, the i-th diagonal element is nonpositive. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doublePOcomputational */

/*  ===================================================================== */
void  qpoequ_(integer *n, quadreal *a, integer *lda, 
	quadreal *s, quadreal *scond, quadreal *amax, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;
    quadreal d__1, d__2;

    /* Local variables */
    integer i__;
    quadreal smin;
    extern void  xerbla_(char *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --s;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -3;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DPOEQU", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	*scond = 1.;
	*amax = 0.;
	return;
    }

/*     Find the minimum and maximum diagonal elements. */

    s[1] = a[a_dim1 + 1];
    smin = s[1];
    *amax = s[1];
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	s[i__] = a[i__ + i__ * a_dim1];
/* Computing MIN */
	d__1 = smin, d__2 = s[i__];
	smin = f2cmin(d__1,d__2);
/* Computing MAX */
	d__1 = *amax, d__2 = s[i__];
	*amax = f2cmax(d__1,d__2);
/* L10: */
    }

    if (smin <= 0.) {

/*        Find the first non-positive diagonal element and return. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (s[i__] <= 0.) {
		*info = i__;
		return;
	    }
/* L20: */
	}
    } else {

/*        Set the scale factors to the reciprocals */
/*        of the diagonal elements. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    s[i__] = 1. / M(sqrt)(s[i__]);
/* L30: */
	}

/*        Compute SCOND = f2cmin(S(I)) / f2cmax(S(I)) */

	*scond = M(sqrt)(smin) / M(sqrt)(*amax);
    }
    return;

/*     End of DPOEQU */

} /* qpoequ_ */

