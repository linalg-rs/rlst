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
static integer c_n1 = -1;
static real c_b21 = 1.f;

/* > \brief \b CLAUUM computes the product UUH or LHL, where U and L are upper or lower triangular matrices (b
locked algorithm). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CLAUUM + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/clauum.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/clauum.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/clauum.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CLAUUM( UPLO, N, A, LDA, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, N */
/*       COMPLEX            A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CLAUUM computes the product U * U**H or L**H * L, where the triangular */
/* > factor U or L is stored in the upper or lower triangular part of */
/* > the array A. */
/* > */
/* > If UPLO = 'U' or 'u' then the upper triangle of the result is stored, */
/* > overwriting the factor U in A. */
/* > If UPLO = 'L' or 'l' then the lower triangle of the result is stored, */
/* > overwriting the factor L in A. */
/* > */
/* > This is the blocked form of the algorithm, calling Level 3 BLAS. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the triangular factor stored in the array A */
/* >          is upper or lower triangular: */
/* >          = 'U':  Upper triangular */
/* >          = 'L':  Lower triangular */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the triangular factor U or L.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          On entry, the triangular factor U or L. */
/* >          On exit, if UPLO = 'U', the upper triangle of A is */
/* >          overwritten with the upper triangle of the product U * U**H; */
/* >          if UPLO = 'L', the lower triangle of A is overwritten with */
/* >          the lower triangle of the product L**H * L. */
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

/* > \ingroup complexOTHERauxiliary */

/*  ===================================================================== */
void  clauum_(char *uplo, integer *n, complex *a, integer *lda,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer i__, ib, nb;
    extern void  cgemm_(char *, char *, integer *, integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, complex *, integer *), cherk_(char *, 
	    char *, integer *, integer *, real *, complex *, integer *, real *
	    , complex *, integer *);
    extern logical lsame_(char *, char *);
    extern void  ctrmm_(char *, char *, char *, char *, 
	    integer *, integer *, complex *, complex *, integer *, complex *, 
	    integer *);
    logical upper;
    extern void  clauu2_(char *, integer *, complex *, integer 
	    *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
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
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CLAUUM", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "CLAUUM", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);

    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	clauu2_(uplo, n, &a[a_offset], lda, info);
    } else {

/*        Use blocked code */

	if (upper) {

/*           Compute the product U * U**H. */

	    i__1 = *n;
	    i__2 = nb;
	    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = f2cmin(i__3,i__4);
		i__3 = i__ - 1;
		ctrmm_("Right", "Upper", "Conjugate transpose", "Non-unit", &
			i__3, &ib, &c_b1, &a[i__ + i__ * a_dim1], lda, &a[i__ 
			* a_dim1 + 1], lda);
		clauu2_("Upper", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    cgemm_("No transpose", "Conjugate transpose", &i__3, &ib, 
			    &i__4, &c_b1, &a[(i__ + ib) * a_dim1 + 1], lda, &
			    a[i__ + (i__ + ib) * a_dim1], lda, &c_b1, &a[i__ *
			     a_dim1 + 1], lda);
		    i__3 = *n - i__ - ib + 1;
		    cherk_("Upper", "No transpose", &ib, &i__3, &c_b21, &a[
			    i__ + (i__ + ib) * a_dim1], lda, &c_b21, &a[i__ + 
			    i__ * a_dim1], lda);
		}
/* L10: */
	    }
	} else {

/*           Compute the product L**H * L. */

	    i__2 = *n;
	    i__1 = nb;
	    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = f2cmin(i__3,i__4);
		i__3 = i__ - 1;
		ctrmm_("Left", "Lower", "Conjugate transpose", "Non-unit", &
			ib, &i__3, &c_b1, &a[i__ + i__ * a_dim1], lda, &a[i__ 
			+ a_dim1], lda);
		clauu2_("Lower", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    cgemm_("Conjugate transpose", "No transpose", &ib, &i__3, 
			    &i__4, &c_b1, &a[i__ + ib + i__ * a_dim1], lda, &
			    a[i__ + ib + a_dim1], lda, &c_b1, &a[i__ + a_dim1]
			    , lda);
		    i__3 = *n - i__ - ib + 1;
		    cherk_("Lower", "Conjugate transpose", &ib, &i__3, &c_b21,
			     &a[i__ + ib + i__ * a_dim1], lda, &c_b21, &a[i__ 
			    + i__ * a_dim1], lda);
		}
/* L20: */
	    }
	}
    }

    return;

/*     End of CLAUUM */

} /* clauum_ */

