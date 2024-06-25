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

/* > \brief <b> CPBSVX computes the solution to system of linear equations A * X = B for OTHER matrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CPBSVX + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cpbsvx.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cpbsvx.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cpbsvx.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CPBSVX( FACT, UPLO, N, KD, NRHS, AB, LDAB, AFB, LDAFB, */
/*                          EQUED, S, B, LDB, X, LDX, RCOND, FERR, BERR, */
/*                          WORK, RWORK, INFO ) */

/*       CHARACTER          EQUED, FACT, UPLO */
/*       INTEGER            INFO, KD, LDAB, LDAFB, LDB, LDX, N, NRHS */
/*       REAL               RCOND */
/*       REAL               BERR( * ), FERR( * ), RWORK( * ), S( * ) */
/*       COMPLEX            AB( LDAB, * ), AFB( LDAFB, * ), B( LDB, * ), */
/*      $                   WORK( * ), X( LDX, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CPBSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to */
/* > compute the solution to a complex system of linear equations */
/* >    A * X = B, */
/* > where A is an N-by-N Hermitian positive definite band matrix and X */
/* > and B are N-by-NRHS matrices. */
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
/* > 1. If FACT = 'E', real scaling factors are computed to equilibrate */
/* >    the system: */
/* >       diag(S) * A * diag(S) * inv(diag(S)) * X = diag(S) * B */
/* >    Whether or not the system will be equilibrated depends on the */
/* >    scaling of the matrix A, but if equilibration is used, A is */
/* >    overwritten by diag(S)*A*diag(S) and B by diag(S)*B. */
/* > */
/* > 2. If FACT = 'N' or 'E', the Cholesky decomposition is used to */
/* >    factor the matrix A (after equilibration if FACT = 'E') as */
/* >       A = U**H * U,  if UPLO = 'U', or */
/* >       A = L * L**H,  if UPLO = 'L', */
/* >    where U is an upper triangular band matrix, and L is a lower */
/* >    triangular band matrix. */
/* > */
/* > 3. If the leading i-by-i principal minor is not positive definite, */
/* >    then the routine returns with INFO = i. Otherwise, the factored */
/* >    form of A is used to estimate the condition number of the matrix */
/* >    A.  If the reciprocal of the condition number is less than machine */
/* >    precision, INFO = N+1 is returned as a warning, but the routine */
/* >    still goes on to solve for X and compute error bounds as */
/* >    described below. */
/* > */
/* > 4. The system of equations is solved for X using the factored form */
/* >    of A. */
/* > */
/* > 5. Iterative refinement is applied to improve the computed solution */
/* >    matrix and calculate error bounds and backward error estimates */
/* >    for it. */
/* > */
/* > 6. If equilibration was used, the matrix X is premultiplied by */
/* >    diag(S) so that it solves the original system before */
/* >    equilibration. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] FACT */
/* > \verbatim */
/* >          FACT is CHARACTER*1 */
/* >          Specifies whether or not the factored form of the matrix A is */
/* >          supplied on entry, and if not, whether the matrix A should be */
/* >          equilibrated before it is factored. */
/* >          = 'F':  On entry, AFB contains the factored form of A. */
/* >                  If EQUED = 'Y', the matrix A has been equilibrated */
/* >                  with scaling factors given by S.  AB and AFB will not */
/* >                  be modified. */
/* >          = 'N':  The matrix A will be copied to AFB and factored. */
/* >          = 'E':  The matrix A will be equilibrated if necessary, then */
/* >                  copied to AFB and factored. */
/* > \endverbatim */
/* > */
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
/* >          The number of linear equations, i.e., the order of the */
/* >          matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] KD */
/* > \verbatim */
/* >          KD is INTEGER */
/* >          The number of superdiagonals of the matrix A if UPLO = 'U', */
/* >          or the number of subdiagonals if UPLO = 'L'.  KD >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right-hand sides, i.e., the number of columns */
/* >          of the matrices B and X.  NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] AB */
/* > \verbatim */
/* >          AB is COMPLEX array, dimension (LDAB,N) */
/* >          On entry, the upper or lower triangle of the Hermitian band */
/* >          matrix A, stored in the first KD+1 rows of the array, except */
/* >          if FACT = 'F' and EQUED = 'Y', then A must contain the */
/* >          equilibrated matrix diag(S)*A*diag(S).  The j-th column of A */
/* >          is stored in the j-th column of the array AB as follows: */
/* >          if UPLO = 'U', AB(KD+1+i-j,j) = A(i,j) for f2cmax(1,j-KD)<=i<=j; */
/* >          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=f2cmin(N,j+KD). */
/* >          See below for further details. */
/* > */
/* >          On exit, if FACT = 'E' and EQUED = 'Y', A is overwritten by */
/* >          diag(S)*A*diag(S). */
/* > \endverbatim */
/* > */
/* > \param[in] LDAB */
/* > \verbatim */
/* >          LDAB is INTEGER */
/* >          The leading dimension of the array A.  LDAB >= KD+1. */
/* > \endverbatim */
/* > */
/* > \param[in,out] AFB */
/* > \verbatim */
/* >          AFB is COMPLEX array, dimension (LDAFB,N) */
/* >          If FACT = 'F', then AFB is an input argument and on entry */
/* >          contains the triangular factor U or L from the Cholesky */
/* >          factorization A = U**H*U or A = L*L**H of the band matrix */
/* >          A, in the same storage format as A (see AB).  If EQUED = 'Y', */
/* >          then AFB is the factored form of the equilibrated matrix A. */
/* > */
/* >          If FACT = 'N', then AFB is an output argument and on exit */
/* >          returns the triangular factor U or L from the Cholesky */
/* >          factorization A = U**H*U or A = L*L**H. */
/* > */
/* >          If FACT = 'E', then AFB is an output argument and on exit */
/* >          returns the triangular factor U or L from the Cholesky */
/* >          factorization A = U**H*U or A = L*L**H of the equilibrated */
/* >          matrix A (see the description of A for the form of the */
/* >          equilibrated matrix). */
/* > \endverbatim */
/* > */
/* > \param[in] LDAFB */
/* > \verbatim */
/* >          LDAFB is INTEGER */
/* >          The leading dimension of the array AFB.  LDAFB >= KD+1. */
/* > \endverbatim */
/* > */
/* > \param[in,out] EQUED */
/* > \verbatim */
/* >          EQUED is CHARACTER*1 */
/* >          Specifies the form of equilibration that was done. */
/* >          = 'N':  No equilibration (always true if FACT = 'N'). */
/* >          = 'Y':  Equilibration was done, i.e., A has been replaced by */
/* >                  diag(S) * A * diag(S). */
/* >          EQUED is an input argument if FACT = 'F'; otherwise, it is an */
/* >          output argument. */
/* > \endverbatim */
/* > */
/* > \param[in,out] S */
/* > \verbatim */
/* >          S is REAL array, dimension (N) */
/* >          The scale factors for A; not accessed if EQUED = 'N'.  S is */
/* >          an input argument if FACT = 'F'; otherwise, S is an output */
/* >          argument.  If FACT = 'F' and EQUED = 'Y', each element of S */
/* >          must be positive. */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX array, dimension (LDB,NRHS) */
/* >          On entry, the N-by-NRHS right hand side matrix B. */
/* >          On exit, if EQUED = 'N', B is not modified; if EQUED = 'Y', */
/* >          B is overwritten by diag(S) * B. */
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
/* >          X is COMPLEX array, dimension (LDX,NRHS) */
/* >          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X to */
/* >          the original system of equations.  Note that if EQUED = 'Y', */
/* >          A and B are modified on exit, and the solution to the */
/* >          equilibrated system is inv(diag(S))*X. */
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
/* >          RCOND is REAL */
/* >          The estimate of the reciprocal condition number of the matrix */
/* >          A after equilibration (if done).  If RCOND is less than the */
/* >          machine precision (in particular, if RCOND = 0), the matrix */
/* >          is singular to working precision.  This condition is */
/* >          indicated by a return code of INFO > 0. */
/* > \endverbatim */
/* > */
/* > \param[out] FERR */
/* > \verbatim */
/* >          FERR is REAL array, dimension (NRHS) */
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
/* >          BERR is REAL array, dimension (NRHS) */
/* >          The componentwise relative backward error of each solution */
/* >          vector X(j) (i.e., the smallest relative change in */
/* >          any element of A or B that makes X(j) an exact solution). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is REAL array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* >          > 0: if INFO = i, and i is */
/* >                <= N:  the leading minor of order i of A is */
/* >                       not positive definite, so the factorization */
/* >                       could not be completed, and the solution has not */
/* >                       been computed. RCOND = 0 is returned. */
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

/* > \date April 2012 */

/* > \ingroup complexOTHERsolve */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The band storage scheme is illustrated by the following example, when */
/* >  N = 6, KD = 2, and UPLO = 'U': */
/* > */
/* >  Two-dimensional storage of the Hermitian matrix A: */
/* > */
/* >     a11  a12  a13 */
/* >          a22  a23  a24 */
/* >               a33  a34  a35 */
/* >                    a44  a45  a46 */
/* >                         a55  a56 */
/* >     (aij=conjg(aji))         a66 */
/* > */
/* >  Band storage of the upper triangle of A: */
/* > */
/* >      *    *   a13  a24  a35  a46 */
/* >      *   a12  a23  a34  a45  a56 */
/* >     a11  a22  a33  a44  a55  a66 */
/* > */
/* >  Similarly, if UPLO = 'L' the format of A is as follows: */
/* > */
/* >     a11  a22  a33  a44  a55  a66 */
/* >     a21  a32  a43  a54  a65   * */
/* >     a31  a42  a53  a64   *    * */
/* > */
/* >  Array elements marked * are not used by the routine. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  cpbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, complex *ab, integer *ldab, complex *afb, integer *
	ldafb, char *equed, real *s, complex *b, integer *ldb, complex *x, 
	integer *ldx, real *rcond, real *ferr, real *berr, complex *work, 
	real *rwork, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, afb_dim1, afb_offset, b_dim1, b_offset, 
	    x_dim1, x_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2;
    complex q__1;

    /* Local variables */
    integer i__, j, j1, j2;
    real amax, smin, smax;
    extern logical lsame_(char *, char *);
    real scond, anorm;
    extern void  ccopy_(integer *, complex *, integer *, 
	    complex *, integer *);
    logical equil, rcequ, upper;
    extern real clanhb_(char *, char *, integer *, integer *, complex *, 
	    integer *, real *);
    extern void  claqhb_(char *, integer *, integer *, complex 
	    *, integer *, real *, real *, real *, char *), 
	    cpbcon_(char *, integer *, integer *, complex *, integer *, real *
	    , real *, complex *, real *, integer *);
    extern real slamch_(char *);
    logical nofact;
    extern void  clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *), xerbla_(char *, 
	    integer *), cpbequ_(char *, integer *, integer *, complex 
	    *, integer *, real *, real *, real *, integer *), cpbrfs_(
	    char *, integer *, integer *, integer *, complex *, integer *, 
	    complex *, integer *, complex *, integer *, complex *, integer *, 
	    real *, real *, complex *, real *, integer *);
    real bignum;
    extern void  cpbtrf_(char *, integer *, integer *, complex 
	    *, integer *, integer *);
    integer infequ;
    extern void  cpbtrs_(char *, integer *, integer *, integer 
	    *, complex *, integer *, complex *, integer *, integer *);
    real smlnum;


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     April 2012 */


/*  ===================================================================== */


    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    afb_dim1 = *ldafb;
    afb_offset = 1 + afb_dim1;
    afb -= afb_offset;
    --s;
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
    equil = lsame_(fact, "E");
    upper = lsame_(uplo, "U");
    if (nofact || equil) {
	*(unsigned char *)equed = 'N';
	rcequ = FALSE_;
    } else {
	rcequ = lsame_(equed, "Y");
	smlnum = slamch_("Safe minimum");
	bignum = 1.f / smlnum;
    }

/*     Test the input parameters. */

    if (! nofact && ! equil && ! lsame_(fact, "F")) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*kd < 0) {
	*info = -4;
    } else if (*nrhs < 0) {
	*info = -5;
    } else if (*ldab < *kd + 1) {
	*info = -7;
    } else if (*ldafb < *kd + 1) {
	*info = -9;
    } else if (lsame_(fact, "F") && ! (rcequ || lsame_(
	    equed, "N"))) {
	*info = -10;
    } else {
	if (rcequ) {
	    smin = bignum;
	    smax = 0.f;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		r__1 = smin, r__2 = s[j];
		smin = f2cmin(r__1,r__2);
/* Computing MAX */
		r__1 = smax, r__2 = s[j];
		smax = f2cmax(r__1,r__2);
/* L10: */
	    }
	    if (smin <= 0.f) {
		*info = -11;
	    } else if (*n > 0) {
		scond = f2cmax(smin,smlnum) / f2cmin(smax,bignum);
	    } else {
		scond = 1.f;
	    }
	}
	if (*info == 0) {
	    if (*ldb < f2cmax(1,*n)) {
		*info = -13;
	    } else if (*ldx < f2cmax(1,*n)) {
		*info = -15;
	    }
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CPBSVX", &i__1);
	return;
    }

    if (equil) {

/*        Compute row and column scalings to equilibrate the matrix A. */

	cpbequ_(uplo, n, kd, &ab[ab_offset], ldab, &s[1], &scond, &amax, &
		infequ);
	if (infequ == 0) {

/*           Equilibrate the matrix. */

	    claqhb_(uplo, n, kd, &ab[ab_offset], ldab, &s[1], &scond, &amax, 
		    equed);
	    rcequ = lsame_(equed, "Y");
	}
    }

/*     Scale the right-hand side. */

    if (rcequ) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__;
		i__5 = i__ + j * b_dim1;
		q__1.r = s[i__4] * b[i__5].r, q__1.i = s[i__4] * b[i__5].i;
		b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L20: */
	    }
/* L30: */
	}
    }

    if (nofact || equil) {

/*        Compute the Cholesky factorization A = U**H *U or A = L*L**H. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
		i__2 = j - *kd;
		j1 = f2cmax(i__2,1);
		i__2 = j - j1 + 1;
		ccopy_(&i__2, &ab[*kd + 1 - j + j1 + j * ab_dim1], &c__1, &
			afb[*kd + 1 - j + j1 + j * afb_dim1], &c__1);
/* L40: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		i__2 = j + *kd;
		j2 = f2cmin(i__2,*n);
		i__2 = j2 - j + 1;
		ccopy_(&i__2, &ab[j * ab_dim1 + 1], &c__1, &afb[j * afb_dim1 
			+ 1], &c__1);
/* L50: */
	    }
	}

	cpbtrf_(uplo, n, kd, &afb[afb_offset], ldafb, info);

/*        Return if INFO is non-zero. */

	if (*info > 0) {
	    *rcond = 0.f;
	    return;
	}
    }

/*     Compute the norm of the matrix A. */

    anorm = clanhb_("1", uplo, n, kd, &ab[ab_offset], ldab, &rwork[1]);

/*     Compute the reciprocal of the condition number of A. */

    cpbcon_(uplo, n, kd, &afb[afb_offset], ldafb, &anorm, rcond, &work[1], &
	    rwork[1], info);

/*     Compute the solution matrix X. */

    clacpy_("Full", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    cpbtrs_(uplo, n, kd, nrhs, &afb[afb_offset], ldafb, &x[x_offset], ldx, 
	    info);

/*     Use iterative refinement to improve the computed solution and */
/*     compute error bounds and backward error estimates for it. */

    cpbrfs_(uplo, n, kd, nrhs, &ab[ab_offset], ldab, &afb[afb_offset], ldafb, 
	    &b[b_offset], ldb, &x[x_offset], ldx, &ferr[1], &berr[1], &work[1]
	    , &rwork[1], info);

/*     Transform the solution matrix X to a solution of the original */
/*     system. */

    if (rcequ) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * x_dim1;
		i__4 = i__;
		i__5 = i__ + j * x_dim1;
		q__1.r = s[i__4] * x[i__5].r, q__1.i = s[i__4] * x[i__5].i;
		x[i__3].r = q__1.r, x[i__3].i = q__1.i;
/* L60: */
	    }
/* L70: */
	}
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    ferr[j] /= scond;
/* L80: */
	}
    }

/*     Set INFO = N+1 if the matrix is singular to working precision. */

    if (*rcond < slamch_("Epsilon")) {
	*info = *n + 1;
    }

    return;

/*     End of CPBSVX */

} /* cpbsvx_ */
