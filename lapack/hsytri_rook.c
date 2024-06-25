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
static halfreal c_b11 = -1.;
static halfreal c_b13 = 0.;

/* > \brief \b DSYTRI_ROOK */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DSYTRI_ROOK + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsytri_
rook.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsytri_
rook.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsytri_
rook.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DSYTRI_ROOK( UPLO, N, A, LDA, IPIV, WORK, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, N */
/*       INTEGER            IPIV( * ) */
/*       DOUBLE PRECISION   A( LDA, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DSYTRI_ROOK computes the inverse of a doublereal symmetric */
/* > matrix A using the factorization A = U*D*U**T or A = L*D*L**T */
/* > computed by DSYTRF_ROOK. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the details of the factorization are stored */
/* >          as an upper or lower triangular matrix. */
/* >          = 'U':  Upper triangular, form is A = U*D*U**T; */
/* >          = 'L':  Lower triangular, form is A = L*D*L**T. */
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
/* >          A is DOUBLE PRECISION array, dimension (LDA,N) */
/* >          On entry, the block diagonal matrix D and the multipliers */
/* >          used to obtain the factor U or L as computed by DSYTRF_ROOK. */
/* > */
/* >          On exit, if INFO = 0, the (symmetric) inverse of the original */
/* >          matrix.  If UPLO = 'U', the upper triangular part of the */
/* >          inverse is formed and the part of A below the diagonal is not */
/* >          referenced; if UPLO = 'L' the lower triangular part of the */
/* >          inverse is formed and the part of A above the diagonal is */
/* >          not referenced. */
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
/* >          Details of the interchanges and the block structure of D */
/* >          as determined by DSYTRF_ROOK. */
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
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* >          > 0: if INFO = i, D(i,i) = 0; the matrix is singular and its */
/* >               inverse could not be computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date April 2012 */

/* > \ingroup doubleSYcomputational */

/* > \par Contributors: */
/*  ================== */
/* > */
/* > \verbatim */
/* > */
/* >   April 2012, Igor Kozachenko, */
/* >                  Computer Science Division, */
/* >                  University of California, Berkeley */
/* > */
/* >  September 2007, Sven Hammarling, Nicholas J. Higham, Craig Lucas, */
/* >                  School of Mathematics, */
/* >                  University of Manchester */
/* > */
/* > \endverbatim */

/*  ===================================================================== */
void  dsytri_rook__(char *uplo, integer *n, halfreal *a, 
	integer *lda, integer *ipiv, halfreal *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;
    halfreal d__1;

    /* Local variables */
    halfreal d__;
    integer k;
    halfreal t, ak;
    integer kp;
    halfreal akp1;
    extern halfreal hdot_(integer *, halfreal *, integer *, halfreal *, 
	    integer *);
    halfreal temp, akkp1;
    extern logical lsame_(char *, char *);
    extern void  hcopy_(integer *, halfreal *, integer *, 
	    halfreal *, integer *), hswap_(integer *, halfreal *, integer 
	    *, halfreal *, integer *);
    integer kstep;
    logical upper;
    extern void  hsymv_(char *, integer *, halfreal *, 
	    halfreal *, integer *, halfreal *, integer *, halfreal *, 
	    halfreal *, integer *), xerbla_(char *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     April 2012 */


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
	xerbla_("DSYTRI_ROOK", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Check that the diagonal matrix D is nonsingular. */

    if (upper) {

/*        Upper triangular storage: examine D from bottom to top */

	for (*info = *n; *info >= 1; --(*info)) {
	    if (ipiv[*info] > 0 && a[*info + *info * a_dim1] == 0.) {
		return;
	    }
/* L10: */
	}
    } else {

/*        Lower triangular storage: examine D from top to bottom. */

	i__1 = *n;
	for (*info = 1; *info <= i__1; ++(*info)) {
	    if (ipiv[*info] > 0 && a[*info + *info * a_dim1] == 0.) {
		return;
	    }
/* L20: */
	}
    }
    *info = 0;

    if (upper) {

/*        Compute inv(A) from the factorization A = U*D*U**T. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = 1;
L30:

/*        If K > N, exit from loop. */

	if (k > *n) {
	    goto L40;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Invert the diagonal block. */

	    a[k + k * a_dim1] = 1. / a[k + k * a_dim1];

/*           Compute column K of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		hcopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		hsymv_(uplo, &i__1, &c_b11, &a[a_offset], lda, &work[1], &
			c__1, &c_b13, &a[k * a_dim1 + 1], &c__1);
		i__1 = k - 1;
		a[k + k * a_dim1] -= hdot_(&i__1, &work[1], &c__1, &a[k * 
			a_dim1 + 1], &c__1);
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = (d__1 = a[k + (k + 1) * a_dim1], abs(d__1));
	    ak = a[k + k * a_dim1] / t;
	    akp1 = a[k + 1 + (k + 1) * a_dim1] / t;
	    akkp1 = a[k + (k + 1) * a_dim1] / t;
	    d__ = t * (ak * akp1 - 1.);
	    a[k + k * a_dim1] = akp1 / d__;
	    a[k + 1 + (k + 1) * a_dim1] = ak / d__;
	    a[k + (k + 1) * a_dim1] = -akkp1 / d__;

/*           Compute columns K and K+1 of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		hcopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		hsymv_(uplo, &i__1, &c_b11, &a[a_offset], lda, &work[1], &
			c__1, &c_b13, &a[k * a_dim1 + 1], &c__1);
		i__1 = k - 1;
		a[k + k * a_dim1] -= hdot_(&i__1, &work[1], &c__1, &a[k * 
			a_dim1 + 1], &c__1);
		i__1 = k - 1;
		a[k + (k + 1) * a_dim1] -= hdot_(&i__1, &a[k * a_dim1 + 1], &
			c__1, &a[(k + 1) * a_dim1 + 1], &c__1);
		i__1 = k - 1;
		hcopy_(&i__1, &a[(k + 1) * a_dim1 + 1], &c__1, &work[1], &
			c__1);
		i__1 = k - 1;
		hsymv_(uplo, &i__1, &c_b11, &a[a_offset], lda, &work[1], &
			c__1, &c_b13, &a[(k + 1) * a_dim1 + 1], &c__1);
		i__1 = k - 1;
		a[k + 1 + (k + 1) * a_dim1] -= hdot_(&i__1, &work[1], &c__1, &
			a[(k + 1) * a_dim1 + 1], &c__1);
	    }
	    kstep = 2;
	}

	if (kstep == 1) {

/*           Interchange rows and columns K and IPIV(K) in the leading */
/*           submatrix A(1:k+1,1:k+1) */

	    kp = ipiv[k];
	    if (kp != k) {
		if (kp > 1) {
		    i__1 = kp - 1;
		    hswap_(&i__1, &a[k * a_dim1 + 1], &c__1, &a[kp * a_dim1 + 
			    1], &c__1);
		}
		i__1 = k - kp - 1;
		hswap_(&i__1, &a[kp + 1 + k * a_dim1], &c__1, &a[kp + (kp + 1)
			 * a_dim1], lda);
		temp = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = temp;
	    }
	} else {

/*           Interchange rows and columns K and K+1 with -IPIV(K) and */
/*           -IPIV(K+1)in the leading submatrix A(1:k+1,1:k+1) */

	    kp = -ipiv[k];
	    if (kp != k) {
		if (kp > 1) {
		    i__1 = kp - 1;
		    hswap_(&i__1, &a[k * a_dim1 + 1], &c__1, &a[kp * a_dim1 + 
			    1], &c__1);
		}
		i__1 = k - kp - 1;
		hswap_(&i__1, &a[kp + 1 + k * a_dim1], &c__1, &a[kp + (kp + 1)
			 * a_dim1], lda);

		temp = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = temp;
		temp = a[k + (k + 1) * a_dim1];
		a[k + (k + 1) * a_dim1] = a[kp + (k + 1) * a_dim1];
		a[kp + (k + 1) * a_dim1] = temp;
	    }

	    ++k;
	    kp = -ipiv[k];
	    if (kp != k) {
		if (kp > 1) {
		    i__1 = kp - 1;
		    hswap_(&i__1, &a[k * a_dim1 + 1], &c__1, &a[kp * a_dim1 + 
			    1], &c__1);
		}
		i__1 = k - kp - 1;
		hswap_(&i__1, &a[kp + 1 + k * a_dim1], &c__1, &a[kp + (kp + 1)
			 * a_dim1], lda);
		temp = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = temp;
	    }
	}

	++k;
	goto L30;
L40:

	;
    } else {

/*        Compute inv(A) from the factorization A = L*D*L**T. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = *n;
L50:

/*        If K < 1, exit from loop. */

	if (k < 1) {
	    goto L60;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Invert the diagonal block. */

	    a[k + k * a_dim1] = 1. / a[k + k * a_dim1];

/*           Compute column K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		hcopy_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		hsymv_(uplo, &i__1, &c_b11, &a[k + 1 + (k + 1) * a_dim1], lda,
			 &work[1], &c__1, &c_b13, &a[k + 1 + k * a_dim1], &
			c__1);
		i__1 = *n - k;
		a[k + k * a_dim1] -= hdot_(&i__1, &work[1], &c__1, &a[k + 1 + 
			k * a_dim1], &c__1);
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = (d__1 = a[k + (k - 1) * a_dim1], abs(d__1));
	    ak = a[k - 1 + (k - 1) * a_dim1] / t;
	    akp1 = a[k + k * a_dim1] / t;
	    akkp1 = a[k + (k - 1) * a_dim1] / t;
	    d__ = t * (ak * akp1 - 1.);
	    a[k - 1 + (k - 1) * a_dim1] = akp1 / d__;
	    a[k + k * a_dim1] = ak / d__;
	    a[k + (k - 1) * a_dim1] = -akkp1 / d__;

/*           Compute columns K-1 and K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		hcopy_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		hsymv_(uplo, &i__1, &c_b11, &a[k + 1 + (k + 1) * a_dim1], lda,
			 &work[1], &c__1, &c_b13, &a[k + 1 + k * a_dim1], &
			c__1);
		i__1 = *n - k;
		a[k + k * a_dim1] -= hdot_(&i__1, &work[1], &c__1, &a[k + 1 + 
			k * a_dim1], &c__1);
		i__1 = *n - k;
		a[k + (k - 1) * a_dim1] -= hdot_(&i__1, &a[k + 1 + k * a_dim1]
			, &c__1, &a[k + 1 + (k - 1) * a_dim1], &c__1);
		i__1 = *n - k;
		hcopy_(&i__1, &a[k + 1 + (k - 1) * a_dim1], &c__1, &work[1], &
			c__1);
		i__1 = *n - k;
		hsymv_(uplo, &i__1, &c_b11, &a[k + 1 + (k + 1) * a_dim1], lda,
			 &work[1], &c__1, &c_b13, &a[k + 1 + (k - 1) * a_dim1]
			, &c__1);
		i__1 = *n - k;
		a[k - 1 + (k - 1) * a_dim1] -= hdot_(&i__1, &work[1], &c__1, &
			a[k + 1 + (k - 1) * a_dim1], &c__1);
	    }
	    kstep = 2;
	}

	if (kstep == 1) {

/*           Interchange rows and columns K and IPIV(K) in the trailing */
/*           submatrix A(k-1:n,k-1:n) */

	    kp = ipiv[k];
	    if (kp != k) {
		if (kp < *n) {
		    i__1 = *n - kp;
		    hswap_(&i__1, &a[kp + 1 + k * a_dim1], &c__1, &a[kp + 1 + 
			    kp * a_dim1], &c__1);
		}
		i__1 = kp - k - 1;
		hswap_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &a[kp + (k + 1) *
			 a_dim1], lda);
		temp = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = temp;
	    }
	} else {

/*           Interchange rows and columns K and K-1 with -IPIV(K) and */
/*           -IPIV(K-1) in the trailing submatrix A(k-1:n,k-1:n) */

	    kp = -ipiv[k];
	    if (kp != k) {
		if (kp < *n) {
		    i__1 = *n - kp;
		    hswap_(&i__1, &a[kp + 1 + k * a_dim1], &c__1, &a[kp + 1 + 
			    kp * a_dim1], &c__1);
		}
		i__1 = kp - k - 1;
		hswap_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &a[kp + (k + 1) *
			 a_dim1], lda);

		temp = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = temp;
		temp = a[k + (k - 1) * a_dim1];
		a[k + (k - 1) * a_dim1] = a[kp + (k - 1) * a_dim1];
		a[kp + (k - 1) * a_dim1] = temp;
	    }

	    --k;
	    kp = -ipiv[k];
	    if (kp != k) {
		if (kp < *n) {
		    i__1 = *n - kp;
		    hswap_(&i__1, &a[kp + 1 + k * a_dim1], &c__1, &a[kp + 1 + 
			    kp * a_dim1], &c__1);
		}
		i__1 = kp - k - 1;
		hswap_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &a[kp + (k + 1) *
			 a_dim1], lda);
		temp = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = temp;
	    }
	}

	--k;
	goto L50;
L60:
	;
    }

    return;

/*     End of DSYTRI_ROOK */

} /* dsytri_rook__ */

