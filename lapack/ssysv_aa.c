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

static integer c_n1 = -1;

/* > \brief <b> SSYSV_AA computes the solution to system of linear equations A * X = B for SY matrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SSYSV_AA + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ssysv_a
a.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ssysv_a
a.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ssysv_a
a.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SSYSV_AA( UPLO, N, NRHS, A, LDA, IPIV, B, LDB, WORK, */
/*                            LWORK, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            N, NRHS, LDA, LDB, LWORK, INFO */
/*       INTEGER            IPIV( * ) */
/*       REAL               A( LDA, * ), B( LDB, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SSYSV computes the solution to a real system of linear equations */
/* >    A * X = B, */
/* > where A is an N-by-N symmetric matrix and X and B are N-by-NRHS */
/* > matrices. */
/* > */
/* > Aasen's algorithm is used to factor A as */
/* >    A = U * T * U**T,  if UPLO = 'U', or */
/* >    A = L * T * L**T,  if UPLO = 'L', */
/* > where U (or L) is a product of permutation and unit upper (lower) */
/* > triangular matrices, and T is symmetric tridiagonal. The factored */
/* > form of A is then used to solve the system of equations A * X = B. */
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
/* >          The number of linear equations, i.e., the order of the */
/* >          matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right hand sides, i.e., the number of columns */
/* >          of the matrix B.  NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is REAL array, dimension (LDA,N) */
/* >          On entry, the symmetric matrix A.  If UPLO = 'U', the leading */
/* >          N-by-N upper triangular part of A contains the upper */
/* >          triangular part of the matrix A, and the strictly lower */
/* >          triangular part of A is not referenced.  If UPLO = 'L', the */
/* >          leading N-by-N lower triangular part of A contains the lower */
/* >          triangular part of the matrix A, and the strictly upper */
/* >          triangular part of A is not referenced. */
/* > */
/* >          On exit, if INFO = 0, the tridiagonal matrix T and the */
/* >          multipliers used to obtain the factor U or L from the */
/* >          factorization A = U*T*U**T or A = L*T*L**T as computed by */
/* >          SSYTRF. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          On exit, it contains the details of the interchanges, i.e., */
/* >          the row and column k of A were interchanged with the */
/* >          row and column IPIV(k). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is REAL array, dimension (LDB,NRHS) */
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
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is REAL array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The length of WORK.  LWORK >= MAX(1,2*N,3*N-2), and for */
/* >          the best performance, LWORK >= MAX(1,N*NB), where NB is */
/* >          the optimal blocksize for SSYTRF_AA. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the WORK array, returns */
/* >          this value as the first entry of the WORK array, and no error */
/* >          message related to LWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* >          > 0: if INFO = i, D(i,i) is exactly zero.  The factorization */
/* >               has been completed, but the block diagonal matrix D is */
/* >               exactly singular, so the solution could not be computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2017 */

/* > \ingroup realSYsolve */

/*  ===================================================================== */
void  ssysv_aa__(char *uplo, integer *n, integer *nrhs, real *
	a, integer *lda, integer *ipiv, real *b, integer *ldb, real *work, 
	integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    extern void  ssytrf_aa__(char *, integer *, real *, 
	    integer *, integer *, real *, integer *, integer *), 
	    ssytrs_aa__(char *, integer *, integer *, real *, integer *, 
	    integer *, real *, integer *, real *, integer *, integer *);
    extern logical lsame_(char *, char *);
    integer lwkopt_sytrf__, lwkopt_sytrs__;
    extern void  xerbla_(char *, integer *);
    integer lwkopt;
    logical lquery;


/*  -- LAPACK driver routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -8;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = *n << 1, i__2 = *n * 3 - 2;
	if (*lwork < f2cmax(i__1,i__2) && ! lquery) {
	    *info = -10;
	}
    }

    if (*info == 0) {
	ssytrf_aa__(uplo, n, &a[a_offset], lda, &ipiv[1], &work[1], &c_n1, 
		info);
	lwkopt_sytrf__ = (integer) work[1];
	ssytrs_aa__(uplo, n, nrhs, &a[a_offset], lda, &ipiv[1], &b[b_offset], 
		ldb, &work[1], &c_n1, info);
	lwkopt_sytrs__ = (integer) work[1];
	lwkopt = f2cmax(lwkopt_sytrf__,lwkopt_sytrs__);
	work[1] = (real) lwkopt;
	if (*lwork < lwkopt && ! lquery) {
	    *info = -10;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYSV_AA", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Compute the factorization A = U*T*U**T or A = L*T*L**T. */

    ssytrf_aa__(uplo, n, &a[a_offset], lda, &ipiv[1], &work[1], lwork, info);
    if (*info == 0) {

/*        Solve the system A*X = B, overwriting B with X. */

	ssytrs_aa__(uplo, n, nrhs, &a[a_offset], lda, &ipiv[1], &b[b_offset], 
		ldb, &work[1], lwork, info);

    }

    work[1] = (real) lwkopt;

    return;

/*     End of SSYSV_AA */

} /* ssysv_aa__ */
