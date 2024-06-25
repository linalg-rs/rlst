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

/* > \brief <b> CPTSV computes the solution to system of linear equations A * X = B for PT matrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CPTSV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cptsv.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cptsv.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cptsv.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CPTSV( N, NRHS, D, E, B, LDB, INFO ) */

/*       INTEGER            INFO, LDB, N, NRHS */
/*       REAL               D( * ) */
/*       COMPLEX            B( LDB, * ), E( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CPTSV computes the solution to a complex system of linear equations */
/* > A*X = B, where A is an N-by-N Hermitian positive definite tridiagonal */
/* > matrix, and X and B are N-by-NRHS matrices. */
/* > */
/* > A is factored as A = L*D*L**H, and the factored form of A is then */
/* > used to solve the system of equations. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right hand sides, i.e., the number of columns */
/* >          of the matrix B.  NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] D */
/* > \verbatim */
/* >          D is REAL array, dimension (N) */
/* >          On entry, the n diagonal elements of the tridiagonal matrix */
/* >          A.  On exit, the n diagonal elements of the diagonal matrix */
/* >          D from the factorization A = L*D*L**H. */
/* > \endverbatim */
/* > */
/* > \param[in,out] E */
/* > \verbatim */
/* >          E is COMPLEX array, dimension (N-1) */
/* >          On entry, the (n-1) subdiagonal elements of the tridiagonal */
/* >          matrix A.  On exit, the (n-1) subdiagonal elements of the */
/* >          unit bidiagonal factor L from the L*D*L**H factorization of */
/* >          A.  E can also be regarded as the superdiagonal of the unit */
/* >          bidiagonal factor U from the U**H*D*U factorization of A. */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX array, dimension (LDB,NRHS) */
/* >          On entry, the N-by-NRHS right hand side matrix B. */
/* >          On exit, if INFO = 0, the N-by-NRHS solution matrix X. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, the leading minor of order i is not */
/* >                positive definite, and the solution has not been */
/* >                computed.  The factorization has not been completed */
/* >                unless i = N. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexPTsolve */

/*  ===================================================================== */
void  cptsv_(integer *n, integer *nrhs, real *d__, complex *e, 
	complex *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1;

    /* Local variables */
    extern void  xerbla_(char *, integer *), cpttrf_(
	    integer *, real *, complex *, integer *), cpttrs_(char *, integer 
	    *, integer *, real *, complex *, complex *, integer *, integer *);


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --e;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*nrhs < 0) {
	*info = -2;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CPTSV ", &i__1);
	return;
    }

/*     Compute the L*D*L**H (or U**H*D*U) factorization of A. */

    cpttrf_(n, &d__[1], &e[1], info);
    if (*info == 0) {

/*        Solve the system A*X = B, overwriting B with X. */

	cpttrs_("Lower", n, nrhs, &d__[1], &e[1], &b[b_offset], ldb, info);
    }
    return;

/*     End of CPTSV */

} /* cptsv_ */

