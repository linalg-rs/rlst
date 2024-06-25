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

static complex c_b9 = {1.f,0.f};
static integer c__1 = 1;

/* > \brief \b CHETRS_AA */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CHETRS_AA + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/chetrs_
aa.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/chetrs_
aa.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/chetrs_
aa.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CHETRS_AA( UPLO, N, NRHS, A, LDA, IPIV, B, LDB, */
/*                             WORK, LWORK, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            N, NRHS, LDA, LDB, LWORK, INFO */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX            A( LDA, * ), B( LDB, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CHETRS_AA solves a system of linear equations A*X = B with a complex */
/* > hermitian matrix A using the factorization A = U*T*U**H or */
/* > A = L*T*L**H computed by CHETRF_AA. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the details of the factorization are stored */
/* >          as an upper or lower triangular matrix. */
/* >          = 'U':  Upper triangular, form is A = U*T*U**H; */
/* >          = 'L':  Lower triangular, form is A = L*T*L**H. */
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
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          Details of factors computed by CHETRF_AA. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          Details of the interchanges as computed by CHETRF_AA. */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX array, dimension (LDB,NRHS) */
/* >          On entry, the right hand side matrix B. */
/* >          On exit, the solution matrix X. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE array, dimension (MAX(1,LWORK)) */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER, LWORK >= MAX(1,3*N-2). */
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

/* > \date November 2017 */

/* > \ingroup complexHEcomputational */

/*  ===================================================================== */
void  chetrs_aa__(char *uplo, integer *n, integer *nrhs, 
	complex *a, integer *lda, integer *ipiv, complex *b, integer *ldb, 
	complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    integer k, kp;
    extern logical lsame_(char *, char *);
    extern void  cswap_(integer *, complex *, integer *, 
	    complex *, integer *), cgtsv_(integer *, integer *, complex *, 
	    complex *, complex *, complex *, integer *, integer *), ctrsm_(
	    char *, char *, char *, char *, integer *, integer *, complex *, 
	    complex *, integer *, complex *, integer *);
    logical upper;
    extern void  clacgv_(integer *, complex *, integer *), 
	    clacpy_(char *, integer *, integer *, complex *, integer *, 
	    complex *, integer *), xerbla_(char *, integer *);
    integer lwkopt;
    logical lquery;


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */



/*  ===================================================================== */


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
    upper = lsame_(uplo, "U");
    lquery = *lwork == -1;
    if (! upper && ! lsame_(uplo, "L")) {
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
	i__1 = 1, i__2 = *n * 3 - 2;
	if (*lwork < f2cmax(i__1,i__2) && ! lquery) {
	    *info = -10;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHETRS_AA", &i__1);
	return;
    } else if (lquery) {
	lwkopt = *n * 3 - 2;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
	return;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return;
    }

    if (upper) {

/*        Solve A*X = B, where A = U*T*U**T. */

/*        P**T * B */

	k = 1;
	while(k <= *n) {
	    kp = ipiv[k];
	    if (kp != k) {
		cswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    ++k;
	}

/*        Compute (U \P**T * B) -> B    [ (U \P**T * B) ] */

	i__1 = *n - 1;
	ctrsm_("L", "U", "C", "U", &i__1, nrhs, &c_b9, &a[(a_dim1 << 1) + 1], 
		lda, &b[b_dim1 + 2], ldb);

/*        Compute T \ B -> B   [ T \ (U \P**T * B) ] */

	i__1 = *lda + 1;
	clacpy_("F", &c__1, n, &a[a_dim1 + 1], &i__1, &work[*n], &c__1);
	if (*n > 1) {
	    i__1 = *n - 1;
	    i__2 = *lda + 1;
	    clacpy_("F", &c__1, &i__1, &a[(a_dim1 << 1) + 1], &i__2, &work[*n 
		    * 2], &c__1);
	    i__1 = *n - 1;
	    i__2 = *lda + 1;
	    clacpy_("F", &c__1, &i__1, &a[(a_dim1 << 1) + 1], &i__2, &work[1],
		     &c__1);
	    i__1 = *n - 1;
	    clacgv_(&i__1, &work[1], &c__1);
	}
	cgtsv_(n, nrhs, &work[1], &work[*n], &work[*n * 2], &b[b_offset], ldb,
		 info);

/*        Compute (U**T \ B) -> B   [ U**T \ (T \ (U \P**T * B) ) ] */

	i__1 = *n - 1;
	ctrsm_("L", "U", "N", "U", &i__1, nrhs, &c_b9, &a[(a_dim1 << 1) + 1], 
		lda, &b[b_dim1 + 2], ldb);

/*        Pivot, P * B  [ P * (U**T \ (T \ (U \P**T * B) )) ] */

	k = *n;
	while(k >= 1) {
	    kp = ipiv[k];
	    if (kp != k) {
		cswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    --k;
	}

    } else {

/*        Solve A*X = B, where A = L*T*L**T. */

/*        Pivot, P**T * B */

	k = 1;
	while(k <= *n) {
	    kp = ipiv[k];
	    if (kp != k) {
		cswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    ++k;
	}

/*        Compute (L \P**T * B) -> B    [ (L \P**T * B) ] */

	i__1 = *n - 1;
	ctrsm_("L", "L", "N", "U", &i__1, nrhs, &c_b9, &a[a_dim1 + 2], lda, &
		b[b_dim1 + 2], ldb);

/*        Compute T \ B -> B   [ T \ (L \P**T * B) ] */

	i__1 = *lda + 1;
	clacpy_("F", &c__1, n, &a[a_dim1 + 1], &i__1, &work[*n], &c__1);
	if (*n > 1) {
	    i__1 = *n - 1;
	    i__2 = *lda + 1;
	    clacpy_("F", &c__1, &i__1, &a[a_dim1 + 2], &i__2, &work[1], &c__1);
	    i__1 = *n - 1;
	    i__2 = *lda + 1;
	    clacpy_("F", &c__1, &i__1, &a[a_dim1 + 2], &i__2, &work[*n * 2], &
		    c__1);
	    i__1 = *n - 1;
	    clacgv_(&i__1, &work[*n * 2], &c__1);
	}
	cgtsv_(n, nrhs, &work[1], &work[*n], &work[*n * 2], &b[b_offset], ldb,
		 info);

/*        Compute (L**T \ B) -> B   [ L**T \ (T \ (L \P**T * B) ) ] */

	i__1 = *n - 1;
	ctrsm_("L", "L", "C", "U", &i__1, nrhs, &c_b9, &a[a_dim1 + 2], lda, &
		b[b_dim1 + 2], ldb);

/*        Pivot, P * B  [ P * (L**T \ (T \ (L \P**T * B) )) ] */

	k = *n;
	while(k >= 1) {
	    kp = ipiv[k];
	    if (kp != k) {
		cswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    --k;
	}

    }

    return;

/*     End of CHETRS_AA */

} /* chetrs_aa__ */

