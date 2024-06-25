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

#define __LAPACK_PRECISION_HALF
#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;

/* > \brief <b> ZGTSVX computes the solution to system of linear equations A * X = B for GT matrices </b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGTSVX + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgtsvx.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgtsvx.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgtsvx.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGTSVX( FACT, TRANS, N, NRHS, DL, D, DU, DLF, DF, DUF, */
/*                          DU2, IPIV, B, LDB, X, LDX, RCOND, FERR, BERR, */
/*                          WORK, RWORK, INFO ) */

/*       CHARACTER          FACT, TRANS */
/*       INTEGER            INFO, LDB, LDX, N, NRHS */
/*       DOUBLE PRECISION   RCOND */
/*       INTEGER            IPIV( * ) */
/*       DOUBLE PRECISION   BERR( * ), FERR( * ), RWORK( * ) */
/*       COMPLEX*16         B( LDB, * ), D( * ), DF( * ), DL( * ), */
/*      $                   DLF( * ), DU( * ), DU2( * ), DUF( * ), */
/*      $                   WORK( * ), X( LDX, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGTSVX uses the LU factorization to compute the solution to a complex */
/* > system of linear equations A * X = B, A**T * X = B, or A**H * X = B, */
/* > where A is a tridiagonal matrix of order N and X and B are N-by-NRHS */
/* > matrices. */
/* > */
/* > Error bounds on the solution and a condition estimate are also */
/* > provided. */
/* > \endverbatim */

/* > \par Description: */
/*  ================= */
/* > */
/* > \verbatim */
/* > */
/* > The following steps are performed: */
/* > */
/* > 1. If FACT = 'N', the LU decomposition is used to factor the matrix A */
/* >    as A = L * U, where L is a product of permutation and unit lower */
/* >    bidiagonal matrices and U is upper triangular with nonzeros in */
/* >    only the main diagonal and first two superdiagonals. */
/* > */
/* > 2. If some U(i,i)=0, so that U is exactly singular, then the routine */
/* >    returns with INFO = i. Otherwise, the factored form of A is used */
/* >    to estimate the condition number of the matrix A.  If the */
/* >    reciprocal of the condition number is less than machine precision, */
/* >    INFO = N+1 is returned as a warning, but the routine still goes on */
/* >    to solve for X and compute error bounds as described below. */
/* > */
/* > 3. The system of equations is solved for X using the factored form */
/* >    of A. */
/* > */
/* > 4. Iterative refinement is applied to improve the computed solution */
/* >    matrix and calculate error bounds and backward error estimates */
/* >    for it. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] FACT */
/* > \verbatim */
/* >          FACT is CHARACTER*1 */
/* >          Specifies whether or not the factored form of A has been */
/* >          supplied on entry. */
/* >          = 'F':  DLF, DF, DUF, DU2, and IPIV contain the factored form */
/* >                  of A; DL, D, DU, DLF, DF, DUF, DU2 and IPIV will not */
/* >                  be modified. */
/* >          = 'N':  The matrix will be copied to DLF, DF, and DUF */
/* >                  and factored. */
/* > \endverbatim */
/* > */
/* > \param[in] TRANS */
/* > \verbatim */
/* >          TRANS is CHARACTER*1 */
/* >          Specifies the form of the system of equations: */
/* >          = 'N':  A * X = B     (No transpose) */
/* >          = 'T':  A**T * X = B  (Transpose) */
/* >          = 'C':  A**H * X = B  (Conjugate transpose) */
/* > \endverbatim */
/* > */
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
/* > \param[in] DL */
/* > \verbatim */
/* >          DL is COMPLEX*16 array, dimension (N-1) */
/* >          The (n-1) subdiagonal elements of A. */
/* > \endverbatim */
/* > */
/* > \param[in] D */
/* > \verbatim */
/* >          D is COMPLEX*16 array, dimension (N) */
/* >          The n diagonal elements of A. */
/* > \endverbatim */
/* > */
/* > \param[in] DU */
/* > \verbatim */
/* >          DU is COMPLEX*16 array, dimension (N-1) */
/* >          The (n-1) superdiagonal elements of A. */
/* > \endverbatim */
/* > */
/* > \param[in,out] DLF */
/* > \verbatim */
/* >          DLF is COMPLEX*16 array, dimension (N-1) */
/* >          If FACT = 'F', then DLF is an input argument and on entry */
/* >          contains the (n-1) multipliers that define the matrix L from */
/* >          the LU factorization of A as computed by ZGTTRF. */
/* > */
/* >          If FACT = 'N', then DLF is an output argument and on exit */
/* >          contains the (n-1) multipliers that define the matrix L from */
/* >          the LU factorization of A. */
/* > \endverbatim */
/* > */
/* > \param[in,out] DF */
/* > \verbatim */
/* >          DF is COMPLEX*16 array, dimension (N) */
/* >          If FACT = 'F', then DF is an input argument and on entry */
/* >          contains the n diagonal elements of the upper triangular */
/* >          matrix U from the LU factorization of A. */
/* > */
/* >          If FACT = 'N', then DF is an output argument and on exit */
/* >          contains the n diagonal elements of the upper triangular */
/* >          matrix U from the LU factorization of A. */
/* > \endverbatim */
/* > */
/* > \param[in,out] DUF */
/* > \verbatim */
/* >          DUF is COMPLEX*16 array, dimension (N-1) */
/* >          If FACT = 'F', then DUF is an input argument and on entry */
/* >          contains the (n-1) elements of the first superdiagonal of U. */
/* > */
/* >          If FACT = 'N', then DUF is an output argument and on exit */
/* >          contains the (n-1) elements of the first superdiagonal of U. */
/* > \endverbatim */
/* > */
/* > \param[in,out] DU2 */
/* > \verbatim */
/* >          DU2 is COMPLEX*16 array, dimension (N-2) */
/* >          If FACT = 'F', then DU2 is an input argument and on entry */
/* >          contains the (n-2) elements of the second superdiagonal of */
/* >          U. */
/* > */
/* >          If FACT = 'N', then DU2 is an output argument and on exit */
/* >          contains the (n-2) elements of the second superdiagonal of */
/* >          U. */
/* > \endverbatim */
/* > */
/* > \param[in,out] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          If FACT = 'F', then IPIV is an input argument and on entry */
/* >          contains the pivot indices from the LU factorization of A as */
/* >          computed by ZGTTRF. */
/* > */
/* >          If FACT = 'N', then IPIV is an output argument and on exit */
/* >          contains the pivot indices from the LU factorization of A; */
/* >          row i of the matrix was interchanged with row IPIV(i). */
/* >          IPIV(i) will always be either i or i+1; IPIV(i) = i indicates */
/* >          a row interchange was not required. */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,NRHS) */
/* >          The N-by-NRHS right hand side matrix B. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] X */
/* > \verbatim */
/* >          X is COMPLEX*16 array, dimension (LDX,NRHS) */
/* >          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X. */
/* > \endverbatim */
/* > */
/* > \param[in] LDX */
/* > \verbatim */
/* >          LDX is INTEGER */
/* >          The leading dimension of the array X.  LDX >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] RCOND */
/* > \verbatim */
/* >          RCOND is DOUBLE PRECISION */
/* >          The estimate of the reciprocal condition number of the matrix */
/* >          A.  If RCOND is less than the machine precision (in */
/* >          particular, if RCOND = 0), the matrix is singular to working */
/* >          precision.  This condition is indicated by a return code of */
/* >          INFO > 0. */
/* > \endverbatim */
/* > */
/* > \param[out] FERR */
/* > \verbatim */
/* >          FERR is DOUBLE PRECISION array, dimension (NRHS) */
/* >          The estimated forward error bound for each solution vector */
/* >          X(j) (the j-th column of the solution matrix X). */
/* >          If XTRUE is the true solution corresponding to X(j), FERR(j) */
/* >          is an estimated upper bound for the magnitude of the largest */
/* >          element in (X(j) - XTRUE) divided by the magnitude of the */
/* >          largest element in X(j).  The estimate is as reliable as */
/* >          the estimate for RCOND, and is almost always a slight */
/* >          overestimate of the true error. */
/* > \endverbatim */
/* > */
/* > \param[out] BERR */
/* > \verbatim */
/* >          BERR is DOUBLE PRECISION array, dimension (NRHS) */
/* >          The componentwise relative backward error of each solution */
/* >          vector X(j) (i.e., the smallest relative change in */
/* >          any element of A or B that makes X(j) an exact solution). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, and i is */
/* >                <= N:  U(i,i) is exactly zero.  The factorization */
/* >                       has not been completed unless i = N, but the */
/* >                       factor U is exactly singular, so the solution */
/* >                       and error bounds could not be computed. */
/* >                       RCOND = 0 is returned. */
/* >                = N+1: U is nonsingular, but RCOND is less than machine */
/* >                       precision, meaning that the matrix is singular */
/* >                       to working precision.  Nevertheless, the */
/* >                       solution and error bounds are computed because */
/* >                       there are a number of situations where the */
/* >                       computed solution can be more accurate than the */
/* >                       value of RCOND would suggest. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16GTsolve */

/*  ===================================================================== */
void  kgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, halfcomplex *dl, halfcomplex *d__, halfcomplex *du, 
	halfcomplex *dlf, halfcomplex *df, halfcomplex *duf, 
	halfcomplex *du2, integer *ipiv, halfcomplex *b, integer *ldb, 
	halfcomplex *x, integer *ldx, halfreal *rcond, halfreal *ferr, 
	halfreal *berr, halfcomplex *work, halfreal *rwork, integer *
	info)
{
    /* System generated locals */
    integer b_dim1, b_offset, x_dim1, x_offset, i__1;

    /* Local variables */
    char norm[1];
    extern logical lsame_(char *, char *);
    halfreal anorm;
    extern void  kcopy_(integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *);
    extern halfreal hlamch_(char *);
    logical nofact;
    extern void  xerbla_(char *, integer *);
    extern halfreal klangt_(char *, integer *, halfcomplex *, 
	    halfcomplex *, halfcomplex *);
    logical notran;
    extern void  klacpy_(char *, integer *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *), 
	    kgtcon_(char *, integer *, halfcomplex *, halfcomplex *, 
	    halfcomplex *, halfcomplex *, integer *, halfreal *, 
	    halfreal *, halfcomplex *, integer *), kgtrfs_(char *,
	     integer *, integer *, halfcomplex *, halfcomplex *, 
	    halfcomplex *, halfcomplex *, halfcomplex *, halfcomplex *
	    , halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfreal *, halfreal *, 
	    halfcomplex *, halfreal *, integer *), kgttrf_(
	    integer *, halfcomplex *, halfcomplex *, halfcomplex *, 
	    halfcomplex *, integer *, integer *), kgttrs_(char *, integer *,
	     integer *, halfcomplex *, halfcomplex *, halfcomplex *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, integer *);


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --dl;
    --d__;
    --du;
    --dlf;
    --df;
    --duf;
    --du2;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --ferr;
    --berr;
    --work;
    --rwork;

    /* Function Body */
    *info = 0;
    nofact = lsame_(fact, "N");
    notran = lsame_(trans, "N");
    if (! nofact && ! lsame_(fact, "F")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T") && ! 
	    lsame_(trans, "C")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 0) {
	*info = -4;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -14;
    } else if (*ldx < f2cmax(1,*n)) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGTSVX", &i__1);
	return;
    }

    if (nofact) {

/*        Compute the LU factorization of A. */

	kcopy_(n, &d__[1], &c__1, &df[1], &c__1);
	if (*n > 1) {
	    i__1 = *n - 1;
	    kcopy_(&i__1, &dl[1], &c__1, &dlf[1], &c__1);
	    i__1 = *n - 1;
	    kcopy_(&i__1, &du[1], &c__1, &duf[1], &c__1);
	}
	kgttrf_(n, &dlf[1], &df[1], &duf[1], &du2[1], &ipiv[1], info);

/*        Return if INFO is non-zero. */

	if (*info > 0) {
	    *rcond = 0.;
	    return;
	}
    }

/*     Compute the norm of the matrix A. */

    if (notran) {
	*(unsigned char *)norm = '1';
    } else {
	*(unsigned char *)norm = 'I';
    }
    anorm = klangt_(norm, n, &dl[1], &d__[1], &du[1]);

/*     Compute the reciprocal of the condition number of A. */

    kgtcon_(norm, n, &dlf[1], &df[1], &duf[1], &du2[1], &ipiv[1], &anorm, 
	    rcond, &work[1], info);

/*     Compute the solution vectors X. */

    klacpy_("Full", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    kgttrs_(trans, n, nrhs, &dlf[1], &df[1], &duf[1], &du2[1], &ipiv[1], &x[
	    x_offset], ldx, info);

/*     Use iterative refinement to improve the computed solutions and */
/*     compute error bounds and backward error estimates for them. */

    kgtrfs_(trans, n, nrhs, &dl[1], &d__[1], &du[1], &dlf[1], &df[1], &duf[1],
	     &du2[1], &ipiv[1], &b[b_offset], ldb, &x[x_offset], ldx, &ferr[1]
	    , &berr[1], &work[1], &rwork[1], info);

/*     Set INFO = N+1 if the matrix is singular to working precision. */

    if (*rcond < hlamch_("Epsilon")) {
	*info = *n + 1;
    }

    return;

/*     End of ZGTSVX */

} /* kgtsvx_ */

