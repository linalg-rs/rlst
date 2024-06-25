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

/* > \brief \b ZSYCONV */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZSYCONV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zsyconv
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zsyconv
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zsyconv
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZSYCONV( UPLO, WAY, N, A, LDA, IPIV, E, INFO ) */

/*       CHARACTER          UPLO, WAY */
/*       INTEGER            INFO, LDA, N */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX*16         A( LDA, * ), E( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZSYCONV converts A given by ZHETRF into L and D or vice-versa. */
/* > Get nondiagonal elements of D (returned in workspace) and */
/* > apply or reverse permutation done in TRF. */
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
/* > \param[in] WAY */
/* > \verbatim */
/* >          WAY is CHARACTER*1 */
/* >          = 'C': Convert */
/* >          = 'R': Revert */
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
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          The block diagonal matrix D and the multipliers used to */
/* >          obtain the factor U or L as computed by ZSYTRF. */
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
/* >          as determined by ZSYTRF. */
/* > \endverbatim */
/* > */
/* > \param[out] E */
/* > \verbatim */
/* >          E is COMPLEX*16 array, dimension (N) */
/* >          E stores the supdiagonal/subdiagonal of the symmetric 1-by-1 */
/* >          or 2-by-2 block diagonal matrix D in LDLT. */
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

/* > \ingroup complex16SYcomputational */

/*  ===================================================================== */
void  ksyconv_(char *uplo, char *way, integer *n, 
	halfcomplex *a, integer *lda, integer *ipiv, halfcomplex *e, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, j, ip;
    halfcomplex temp;
    extern logical lsame_(char *, char *);
    logical upper;
    extern void  xerbla_(char *, integer *);
    logical convert;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */



    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    --e;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    convert = lsame_(way, "C");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! convert && ! lsame_(way, "R")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZSYCONV", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

    if (upper) {

/*        A is UPPER */

	if (convert) {

/*           Convert A (A is upper) */

/*           Convert VALUE */

	    i__ = *n;
	    e[1].r = 0., e[1].i = 0.;
	    while(i__ > 1) {
		if (ipiv[i__] < 0) {
		    i__1 = i__;
		    i__2 = i__ - 1 + i__ * a_dim1;
		    e[i__1].r = a[i__2].r, e[i__1].i = a[i__2].i;
		    i__1 = i__ - 1;
		    e[i__1].r = 0., e[i__1].i = 0.;
		    i__1 = i__ - 1 + i__ * a_dim1;
		    a[i__1].r = 0., a[i__1].i = 0.;
		    --i__;
		} else {
		    i__1 = i__;
		    e[i__1].r = 0., e[i__1].i = 0.;
		}
		--i__;
	    }

/*           Convert PERMUTATIONS */

	    i__ = *n;
	    while(i__ >= 1) {
		if (ipiv[i__] > 0) {
		    ip = ipiv[i__];
		    if (i__ < *n) {
			i__1 = *n;
			for (j = i__ + 1; j <= i__1; ++j) {
			    i__2 = ip + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = ip + j * a_dim1;
			    i__3 = i__ + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = i__ + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
/* L12: */
			}
		    }
		} else {
		    ip = -ipiv[i__];
		    if (i__ < *n) {
			i__1 = *n;
			for (j = i__ + 1; j <= i__1; ++j) {
			    i__2 = ip + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = ip + j * a_dim1;
			    i__3 = i__ - 1 + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = i__ - 1 + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
/* L13: */
			}
		    }
		    --i__;
		}
		--i__;
	    }

	} else {

/*           Revert A (A is upper) */

/*           Revert PERMUTATIONS */

	    i__ = 1;
	    while(i__ <= *n) {
		if (ipiv[i__] > 0) {
		    ip = ipiv[i__];
		    if (i__ < *n) {
			i__1 = *n;
			for (j = i__ + 1; j <= i__1; ++j) {
			    i__2 = ip + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = ip + j * a_dim1;
			    i__3 = i__ + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = i__ + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
			}
		    }
		} else {
		    ip = -ipiv[i__];
		    ++i__;
		    if (i__ < *n) {
			i__1 = *n;
			for (j = i__ + 1; j <= i__1; ++j) {
			    i__2 = ip + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = ip + j * a_dim1;
			    i__3 = i__ - 1 + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = i__ - 1 + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
			}
		    }
		}
		++i__;
	    }

/*           Revert VALUE */

	    i__ = *n;
	    while(i__ > 1) {
		if (ipiv[i__] < 0) {
		    i__1 = i__ - 1 + i__ * a_dim1;
		    i__2 = i__;
		    a[i__1].r = e[i__2].r, a[i__1].i = e[i__2].i;
		    --i__;
		}
		--i__;
	    }
	}

    } else {

/*        A is LOWER */

	if (convert) {

/*           Convert A (A is lower) */

/*           Convert VALUE */

	    i__ = 1;
	    i__1 = *n;
	    e[i__1].r = 0., e[i__1].i = 0.;
	    while(i__ <= *n) {
		if (i__ < *n && ipiv[i__] < 0) {
		    i__1 = i__;
		    i__2 = i__ + 1 + i__ * a_dim1;
		    e[i__1].r = a[i__2].r, e[i__1].i = a[i__2].i;
		    i__1 = i__ + 1;
		    e[i__1].r = 0., e[i__1].i = 0.;
		    i__1 = i__ + 1 + i__ * a_dim1;
		    a[i__1].r = 0., a[i__1].i = 0.;
		    ++i__;
		} else {
		    i__1 = i__;
		    e[i__1].r = 0., e[i__1].i = 0.;
		}
		++i__;
	    }

/*           Convert PERMUTATIONS */

	    i__ = 1;
	    while(i__ <= *n) {
		if (ipiv[i__] > 0) {
		    ip = ipiv[i__];
		    if (i__ > 1) {
			i__1 = i__ - 1;
			for (j = 1; j <= i__1; ++j) {
			    i__2 = ip + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = ip + j * a_dim1;
			    i__3 = i__ + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = i__ + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
/* L22: */
			}
		    }
		} else {
		    ip = -ipiv[i__];
		    if (i__ > 1) {
			i__1 = i__ - 1;
			for (j = 1; j <= i__1; ++j) {
			    i__2 = ip + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = ip + j * a_dim1;
			    i__3 = i__ + 1 + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = i__ + 1 + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
/* L23: */
			}
		    }
		    ++i__;
		}
		++i__;
	    }

	} else {

/*           Revert A (A is lower) */

/*           Revert PERMUTATIONS */

	    i__ = *n;
	    while(i__ >= 1) {
		if (ipiv[i__] > 0) {
		    ip = ipiv[i__];
		    if (i__ > 1) {
			i__1 = i__ - 1;
			for (j = 1; j <= i__1; ++j) {
			    i__2 = i__ + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + j * a_dim1;
			    i__3 = ip + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = ip + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
			}
		    }
		} else {
		    ip = -ipiv[i__];
		    --i__;
		    if (i__ > 1) {
			i__1 = i__ - 1;
			for (j = 1; j <= i__1; ++j) {
			    i__2 = i__ + 1 + j * a_dim1;
			    temp.r = a[i__2].r, temp.i = a[i__2].i;
			    i__2 = i__ + 1 + j * a_dim1;
			    i__3 = ip + j * a_dim1;
			    a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
			    i__2 = ip + j * a_dim1;
			    a[i__2].r = temp.r, a[i__2].i = temp.i;
			}
		    }
		}
		--i__;
	    }

/*           Revert VALUE */

	    i__ = 1;
	    while(i__ <= *n - 1) {
		if (ipiv[i__] < 0) {
		    i__1 = i__ + 1 + i__ * a_dim1;
		    i__2 = i__;
		    a[i__1].r = e[i__2].r, a[i__1].i = e[i__2].i;
		    ++i__;
		}
		++i__;
	    }
	}
    }

    return;

/*     End of ZSYCONV */

} /* ksyconv_ */

