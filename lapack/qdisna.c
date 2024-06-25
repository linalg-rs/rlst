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

/* > \brief \b DDISNA */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DDISNA + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ddisna.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ddisna.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ddisna.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DDISNA( JOB, M, N, D, SEP, INFO ) */

/*       CHARACTER          JOB */
/*       INTEGER            INFO, M, N */
/*       DOUBLE PRECISION   D( * ), SEP( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DDISNA computes the reciprocal condition numbers for the eigenvectors */
/* > of a doublereal symmetric or complex Hermitian matrix or for the left or */
/* > right singular vectors of a general m-by-n matrix. The reciprocal */
/* > condition number is the 'gap' between the corresponding eigenvalue or */
/* > singular value and the nearest other one. */
/* > */
/* > The bound on the error, measured by angle in radians, in the I-th */
/* > computed vector is given by */
/* > */
/* >        DLAMCH( 'E' ) * ( ANORM / SEP( I ) ) */
/* > */
/* > where ANORM = 2-norm(A) = f2cmax( abs( D(j) ) ).  SEP(I) is not allowed */
/* > to be smaller than DLAMCH( 'E' )*ANORM in order to limit the size of */
/* > the error bound. */
/* > */
/* > DDISNA may also be used to compute error bounds for eigenvectors of */
/* > the generalized symmetric definite eigenproblem. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOB */
/* > \verbatim */
/* >          JOB is CHARACTER*1 */
/* >          Specifies for which problem the reciprocal condition numbers */
/* >          should be computed: */
/* >          = 'E':  the eigenvectors of a symmetric/Hermitian matrix; */
/* >          = 'L':  the left singular vectors of a general matrix; */
/* >          = 'R':  the right singular vectors of a general matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix. M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          If JOB = 'L' or 'R', the number of columns of the matrix, */
/* >          in which case N >= 0. Ignored if JOB = 'E'. */
/* > \endverbatim */
/* > */
/* > \param[in] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension (M) if JOB = 'E' */
/* >                              dimension (f2cmin(M,N)) if JOB = 'L' or 'R' */
/* >          The eigenvalues (if JOB = 'E') or singular values (if JOB = */
/* >          'L' or 'R') of the matrix, in either increasing or decreasing */
/* >          order. If singular values, they must be non-negative. */
/* > \endverbatim */
/* > */
/* > \param[out] SEP */
/* > \verbatim */
/* >          SEP is DOUBLE PRECISION array, dimension (M) if JOB = 'E' */
/* >                               dimension (f2cmin(M,N)) if JOB = 'L' or 'R' */
/* >          The reciprocal condition numbers of the vectors. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit. */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup auxOTHERcomputational */

/*  ===================================================================== */
void  qdisna_(char *job, integer *m, integer *n, quadreal *
	d__, quadreal *sep, integer *info)
{
    /* System generated locals */
    integer i__1;
    quadreal d__1, d__2, d__3;

    /* Local variables */
    integer i__, k;
    quadreal eps;
    logical decr, left, incr, sing, eigen;
    extern logical lsame_(char *, char *);
    quadreal anorm;
    logical right;
    extern quadreal qlamch_(char *);
    quadreal oldgap, safmin;
    extern void  xerbla_(char *, integer *);
    quadreal newgap, thresh;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    --sep;
    --d__;

    /* Function Body */
    *info = 0;
    eigen = lsame_(job, "E");
    left = lsame_(job, "L");
    right = lsame_(job, "R");
    sing = left || right;
    if (eigen) {
	k = *m;
    } else if (sing) {
	k = f2cmin(*m,*n);
    }
    if (! eigen && ! sing) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (k < 0) {
	*info = -3;
    } else {
	incr = TRUE_;
	decr = TRUE_;
	i__1 = k - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (incr) {
		incr = incr && d__[i__] <= d__[i__ + 1];
	    }
	    if (decr) {
		decr = decr && d__[i__] >= d__[i__ + 1];
	    }
/* L10: */
	}
	if (sing && k > 0) {
	    if (incr) {
		incr = incr && 0. <= d__[1];
	    }
	    if (decr) {
		decr = decr && d__[k] >= 0.;
	    }
	}
	if (! (incr || decr)) {
	    *info = -4;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DDISNA", &i__1);
	return;
    }

/*     Quick return if possible */

    if (k == 0) {
	return;
    }

/*     Compute reciprocal condition numbers */

    if (k == 1) {
	sep[1] = qlamch_("O");
    } else {
	oldgap = (d__1 = d__[2] - d__[1], abs(d__1));
	sep[1] = oldgap;
	i__1 = k - 1;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    newgap = (d__1 = d__[i__ + 1] - d__[i__], abs(d__1));
	    sep[i__] = f2cmin(oldgap,newgap);
	    oldgap = newgap;
/* L20: */
	}
	sep[k] = oldgap;
    }
    if (sing) {
	if (left && *m > *n || right && *m < *n) {
	    if (incr) {
		sep[1] = f2cmin(sep[1],d__[1]);
	    }
	    if (decr) {
/* Computing MIN */
		d__1 = sep[k], d__2 = d__[k];
		sep[k] = f2cmin(d__1,d__2);
	    }
	}
    }

/*     Ensure that reciprocal condition numbers are not less than */
/*     threshold, in order to limit the size of the error bound */

    eps = qlamch_("E");
    safmin = qlamch_("S");
/* Computing MAX */
    d__2 = abs(d__[1]), d__3 = (d__1 = d__[k], abs(d__1));
    anorm = f2cmax(d__2,d__3);
    if (anorm == 0.) {
	thresh = eps;
    } else {
/* Computing MAX */
	d__1 = eps * anorm;
	thresh = f2cmax(d__1,safmin);
    }
    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	d__1 = sep[i__];
	sep[i__] = f2cmax(d__1,thresh);
/* L30: */
    }

    return;

/*     End of DDISNA */

} /* qdisna_ */

