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

/* Table of constant values */

static integer c__1 = 1;

/* > \brief \b DPTCON */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DPTCON + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dptcon.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dptcon.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dptcon.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DPTCON( N, D, E, ANORM, RCOND, WORK, INFO ) */

/*       INTEGER            INFO, N */
/*       DOUBLE PRECISION   ANORM, RCOND */
/*       DOUBLE PRECISION   D( * ), E( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DPTCON computes the reciprocal of the condition number (in the */
/* > 1-norm) of a real symmetric positive definite tridiagonal matrix */
/* > using the factorization A = L*D*L**T or A = U**T*D*U computed by */
/* > DPTTRF. */
/* > */
/* > Norm(inv(A)) is computed by a direct method, and the reciprocal of */
/* > the condition number is computed as */
/* >              RCOND = 1 / (ANORM * norm(inv(A))). */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension (N) */
/* >          The n diagonal elements of the diagonal matrix D from the */
/* >          factorization of A, as computed by DPTTRF. */
/* > \endverbatim */
/* > */
/* > \param[in] E */
/* > \verbatim */
/* >          E is DOUBLE PRECISION array, dimension (N-1) */
/* >          The (n-1) off-diagonal elements of the unit bidiagonal factor */
/* >          U or L from the factorization of A,  as computed by DPTTRF. */
/* > \endverbatim */
/* > */
/* > \param[in] ANORM */
/* > \verbatim */
/* >          ANORM is DOUBLE PRECISION */
/* >          The 1-norm of the original matrix A. */
/* > \endverbatim */
/* > */
/* > \param[out] RCOND */
/* > \verbatim */
/* >          RCOND is DOUBLE PRECISION */
/* >          The reciprocal of the condition number of the matrix A, */
/* >          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is the */
/* >          1-norm of inv(A) computed in this routine. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doublePTcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The method used is described in Nicholas J. Higham, "Efficient */
/* >  Algorithms for Computing the Condition Number of a Tridiagonal */
/* >  Matrix", SIAM J. Sci. Stat. Comput., Vol. 7, No. 1, January 1986. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  dptcon_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *anorm, doublereal *rcond, doublereal *work, integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    integer i__, ix;
    extern integer idamax_(integer *, doublereal *, integer *);
    extern void  xerbla_(char *, integer *);
    doublereal ainvnm;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments. */

    /* Parameter adjustments */
    --work;
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*anorm < 0.) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DPTCON", &i__1);
	return;
    }

/*     Quick return if possible */

    *rcond = 0.;
    if (*n == 0) {
	*rcond = 1.;
	return;
    } else if (*anorm == 0.) {
	return;
    }

/*     Check that D(1:N) is positive. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (d__[i__] <= 0.) {
	    return;
	}
/* L10: */
    }

/*     Solve M(A) * x = e, where M(A) = (m(i,j)) is given by */

/*        m(i,j) =  abs(A(i,j)), i = j, */
/*        m(i,j) = -abs(A(i,j)), i .ne. j, */

/*     and e = [ 1, 1, ..., 1 ]**T.  Note M(A) = M(L)*D*M(L)**T. */

/*     Solve M(L) * x = e. */

    work[1] = 1.;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	work[i__] = work[i__ - 1] * (d__1 = e[i__ - 1], abs(d__1)) + 1.;
/* L20: */
    }

/*     Solve D * M(L)**T * x = b. */

    work[*n] /= d__[*n];
    for (i__ = *n - 1; i__ >= 1; --i__) {
	work[i__] = work[i__] / d__[i__] + work[i__ + 1] * (d__1 = e[i__], 
		abs(d__1));
/* L30: */
    }

/*     Compute AINVNM = f2cmax(x(i)), 1<=i<=n. */

    ix = idamax_(n, &work[1], &c__1);
    ainvnm = (d__1 = work[ix], abs(d__1));

/*     Compute the reciprocal condition number. */

    if (ainvnm != 0.) {
	*rcond = 1. / ainvnm / *anorm;
    }

    return;

/*     End of DPTCON */

} /* dptcon_ */
