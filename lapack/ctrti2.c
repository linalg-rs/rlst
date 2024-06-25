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

static complex c_b1 = {1.f,0.f};
static integer c__1 = 1;

/* > \brief \b CTRTI2 computes the inverse of a triangular matrix (unblocked algorithm). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CTRTI2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ctrti2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ctrti2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ctrti2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CTRTI2( UPLO, DIAG, N, A, LDA, INFO ) */

/*       CHARACTER          DIAG, UPLO */
/*       INTEGER            INFO, LDA, N */
/*       COMPLEX            A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CTRTI2 computes the inverse of a complex upper or lower triangular */
/* > matrix. */
/* > */
/* > This is the Level 2 BLAS version of the algorithm. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the matrix A is upper or lower triangular. */
/* >          = 'U':  Upper triangular */
/* >          = 'L':  Lower triangular */
/* > \endverbatim */
/* > */
/* > \param[in] DIAG */
/* > \verbatim */
/* >          DIAG is CHARACTER*1 */
/* >          Specifies whether or not the matrix A is unit triangular. */
/* >          = 'N':  Non-unit triangular */
/* >          = 'U':  Unit triangular */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          On entry, the triangular matrix A.  If UPLO = 'U', the */
/* >          leading n by n upper triangular part of the array A contains */
/* >          the upper triangular matrix, and the strictly lower */
/* >          triangular part of A is not referenced.  If UPLO = 'L', the */
/* >          leading n by n lower triangular part of the array A contains */
/* >          the lower triangular matrix, and the strictly upper */
/* >          triangular part of A is not referenced.  If DIAG = 'U', the */
/* >          diagonal elements of A are also not referenced and are */
/* >          assumed to be 1. */
/* > */
/* >          On exit, the (triangular) inverse of the original matrix, in */
/* >          the same storage format. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -k, the k-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexOTHERcomputational */

/*  ===================================================================== */
void  ctrti2_(char *uplo, char *diag, integer *n, complex *a, 
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    complex q__1;

    /* Local variables */
    integer j;
    complex ajj;
    extern void  cscal_(integer *, complex *, complex *, 
	    integer *);
    extern logical lsame_(char *, char *);
    logical upper;
    extern void  ctrmv_(char *, char *, char *, integer *, 
	    complex *, integer *, complex *, integer *), xerbla_(char *, integer *);
    logical nounit;


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

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    nounit = lsame_(diag, "N");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTRTI2", &i__1);
	return;
    }

    if (upper) {

/*        Compute inverse of upper triangular matrix. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (nounit) {
		i__2 = j + j * a_dim1;
		c_div(&q__1, &c_b1, &a[j + j * a_dim1]);
		a[i__2].r = q__1.r, a[i__2].i = q__1.i;
		i__2 = j + j * a_dim1;
		q__1.r = -a[i__2].r, q__1.i = -a[i__2].i;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    } else {
		q__1.r = -1.f, q__1.i = -0.f;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    }

/*           Compute elements 1:j-1 of j-th column. */

	    i__2 = j - 1;
	    ctrmv_("Upper", "No transpose", diag, &i__2, &a[a_offset], lda, &
		    a[j * a_dim1 + 1], &c__1);
	    i__2 = j - 1;
	    cscal_(&i__2, &ajj, &a[j * a_dim1 + 1], &c__1);
/* L10: */
	}
    } else {

/*        Compute inverse of lower triangular matrix. */

	for (j = *n; j >= 1; --j) {
	    if (nounit) {
		i__1 = j + j * a_dim1;
		c_div(&q__1, &c_b1, &a[j + j * a_dim1]);
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
		i__1 = j + j * a_dim1;
		q__1.r = -a[i__1].r, q__1.i = -a[i__1].i;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    } else {
		q__1.r = -1.f, q__1.i = -0.f;
		ajj.r = q__1.r, ajj.i = q__1.i;
	    }
	    if (j < *n) {

/*              Compute elements j+1:n of j-th column. */

		i__1 = *n - j;
		ctrmv_("Lower", "No transpose", diag, &i__1, &a[j + 1 + (j + 
			1) * a_dim1], lda, &a[j + 1 + j * a_dim1], &c__1);
		i__1 = *n - j;
		cscal_(&i__1, &ajj, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }

    return;

/*     End of CTRTI2 */

} /* ctrti2_ */

