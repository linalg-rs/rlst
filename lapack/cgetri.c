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

static complex c_b2 = {1.f,0.f};
static integer c__1 = 1;
static integer c_n1 = -1;
static integer c__2 = 2;

/* > \brief \b CGETRI */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CGETRI + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cgetri.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cgetri.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cgetri.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CGETRI( N, A, LDA, IPIV, WORK, LWORK, INFO ) */

/*       INTEGER            INFO, LDA, LWORK, N */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX            A( LDA, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CGETRI computes the inverse of a matrix using the LU factorization */
/* > computed by CGETRF. */
/* > */
/* > This method inverts U and then computes inv(A) by solving the system */
/* > inv(A)*L = inv(U) for inv(A). */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          On entry, the factors L and U from the factorization */
/* >          A = P*L*U as computed by CGETRF. */
/* >          On exit, if INFO = 0, the inverse of the original matrix A. */
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
/* >          The pivot indices from CGETRF; for 1<=i<=N, row i of the */
/* >          matrix was interchanged with row IPIV(i). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO=0, then WORK(1) returns the optimal LWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK.  LWORK >= f2cmax(1,N). */
/* >          For optimal performance LWORK >= N*NB, where NB is */
/* >          the optimal blocksize returned by ILAENV. */
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
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, U(i,i) is exactly zero; the matrix is */
/* >                singular and its inverse could not be computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexGEcomputational */

/*  ===================================================================== */
void  cgetri_(integer *n, complex *a, integer *lda, integer *
	ipiv, complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1;

    /* Local variables */
    integer i__, j, jb, nb, jj, jp, nn, iws;
    extern void  cgemm_(char *, char *, integer *, integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, complex *, integer *), cgemv_(char *, 
	    integer *, integer *, complex *, complex *, integer *, complex *, 
	    integer *, complex *, complex *, integer *);
    integer nbmin;
    extern void  cswap_(integer *, complex *, integer *, 
	    complex *, integer *), ctrsm_(char *, char *, char *, char *, 
	    integer *, integer *, complex *, complex *, integer *, complex *, 
	    integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern void  xerbla_(char *, integer *);
    integer ldwork;
    extern void  ctrtri_(char *, char *, integer *, complex *, 
	    integer *, integer *);
    integer lwkopt;
    logical lquery;


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
    --ipiv;
    --work;

    /* Function Body */
    *info = 0;
    nb = ilaenv_(&c__1, "CGETRI", " ", n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);
    lwkopt = *n * nb;
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -3;
    } else if (*lwork < f2cmax(1,*n) && ! lquery) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGETRI", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Form inv(U).  If INFO > 0 from CTRTRI, then U is singular, */
/*     and the inverse is not computed. */

    ctrtri_("Upper", "Non-unit", n, &a[a_offset], lda, info);
    if (*info > 0) {
	return;
    }

    nbmin = 2;
    ldwork = *n;
    if (nb > 1 && nb < *n) {
/* Computing MAX */
	i__1 = ldwork * nb;
	iws = f2cmax(i__1,1);
	if (*lwork < iws) {
	    nb = *lwork / ldwork;
/* Computing MAX */
	    i__1 = 2, i__2 = ilaenv_(&c__2, "CGETRI", " ", n, &c_n1, &c_n1, &
		    c_n1, (ftnlen)6, (ftnlen)1);
	    nbmin = f2cmax(i__1,i__2);
	}
    } else {
	iws = *n;
    }

/*     Solve the equation inv(A)*L = inv(U) for inv(A). */

    if (nb < nbmin || nb >= *n) {

/*        Use unblocked code. */

	for (j = *n; j >= 1; --j) {

/*           Copy current column of L to WORK and replace with zeros. */

	    i__1 = *n;
	    for (i__ = j + 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		i__3 = i__ + j * a_dim1;
		work[i__2].r = a[i__3].r, work[i__2].i = a[i__3].i;
		i__2 = i__ + j * a_dim1;
		a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L10: */
	    }

/*           Compute current column of inv(A). */

	    if (j < *n) {
		i__1 = *n - j;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", n, &i__1, &q__1, &a[(j + 1) * a_dim1 + 
			1], lda, &work[j + 1], &c__1, &c_b2, &a[j * a_dim1 + 
			1], &c__1);
	    }
/* L20: */
	}
    } else {

/*        Use blocked code. */

	nn = (*n - 1) / nb * nb + 1;
	i__1 = -nb;
	for (j = nn; i__1 < 0 ? j >= 1 : j <= 1; j += i__1) {
/* Computing MIN */
	    i__2 = nb, i__3 = *n - j + 1;
	    jb = f2cmin(i__2,i__3);

/*           Copy current block column of L to WORK and replace with */
/*           zeros. */

	    i__2 = j + jb - 1;
	    for (jj = j; jj <= i__2; ++jj) {
		i__3 = *n;
		for (i__ = jj + 1; i__ <= i__3; ++i__) {
		    i__4 = i__ + (jj - j) * ldwork;
		    i__5 = i__ + jj * a_dim1;
		    work[i__4].r = a[i__5].r, work[i__4].i = a[i__5].i;
		    i__4 = i__ + jj * a_dim1;
		    a[i__4].r = 0.f, a[i__4].i = 0.f;
/* L30: */
		}
/* L40: */
	    }

/*           Compute current block column of inv(A). */

	    if (j + jb <= *n) {
		i__2 = *n - j - jb + 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemm_("No transpose", "No transpose", n, &jb, &i__2, &q__1, &
			a[(j + jb) * a_dim1 + 1], lda, &work[j + jb], &ldwork,
			 &c_b2, &a[j * a_dim1 + 1], lda);
	    }
	    ctrsm_("Right", "Lower", "No transpose", "Unit", n, &jb, &c_b2, &
		    work[j], &ldwork, &a[j * a_dim1 + 1], lda);
/* L50: */
	}
    }

/*     Apply column interchanges. */

    for (j = *n - 1; j >= 1; --j) {
	jp = ipiv[j];
	if (jp != j) {
	    cswap_(n, &a[j * a_dim1 + 1], &c__1, &a[jp * a_dim1 + 1], &c__1);
	}
/* L60: */
    }

    work[1].r = (real) iws, work[1].i = 0.f;
    return;

/*     End of CGETRI */

} /* cgetri_ */

