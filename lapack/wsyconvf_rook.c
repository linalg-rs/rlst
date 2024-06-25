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

#define __LAPACK_PRECISION_QUAD
#include "f2c.h"

/* > \brief \b ZSYCONVF_ROOK */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZSYCONVF_ROOK + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zsyconv
f_rook.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zsyconv
f_rook.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zsyconv
f_rook.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZSYCONVF_ROOK( UPLO, WAY, N, A, LDA, E, IPIV, INFO ) */

/*       CHARACTER          UPLO, WAY */
/*       INTEGER            INFO, LDA, N */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX*16         A( LDA, * ), E( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > If parameter WAY = 'C': */
/* > ZSYCONVF_ROOK converts the factorization output format used in */
/* > ZSYTRF_ROOK provided on entry in parameter A into the factorization */
/* > output format used in ZSYTRF_RK (or ZSYTRF_BK) that is stored */
/* > on exit in parameters A and E. IPIV format for ZSYTRF_ROOK and */
/* > ZSYTRF_RK (or ZSYTRF_BK) is the same and is not converted. */
/* > */
/* > If parameter WAY = 'R': */
/* > ZSYCONVF_ROOK performs the conversion in reverse direction, i.e. */
/* > converts the factorization output format used in ZSYTRF_RK */
/* > (or ZSYTRF_BK) provided on entry in parameters A and E into */
/* > the factorization output format used in ZSYTRF_ROOK that is stored */
/* > on exit in parameter A. IPIV format for ZSYTRF_ROOK and */
/* > ZSYTRF_RK (or ZSYTRF_BK) is the same and is not converted. */
/* > */
/* > ZSYCONVF_ROOK can also convert in Hermitian matrix case, i.e. between */
/* > formats used in ZHETRF_ROOK and ZHETRF_RK (or ZHETRF_BK). */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the details of the factorization are */
/* >          stored as an upper or lower triangular matrix A. */
/* >          = 'U':  Upper triangular */
/* >          = 'L':  Lower triangular */
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
/* > */
/* >          1) If WAY ='C': */
/* > */
/* >          On entry, contains factorization details in format used in */
/* >          ZSYTRF_ROOK: */
/* >            a) all elements of the symmetric block diagonal */
/* >               matrix D on the diagonal of A and on superdiagonal */
/* >               (or subdiagonal) of A, and */
/* >            b) If UPLO = 'U': multipliers used to obtain factor U */
/* >               in the superdiagonal part of A. */
/* >               If UPLO = 'L': multipliers used to obtain factor L */
/* >               in the superdiagonal part of A. */
/* > */
/* >          On exit, contains factorization details in format used in */
/* >          ZSYTRF_RK or ZSYTRF_BK: */
/* >            a) ONLY diagonal elements of the symmetric block diagonal */
/* >               matrix D on the diagonal of A, i.e. D(k,k) = A(k,k); */
/* >               (superdiagonal (or subdiagonal) elements of D */
/* >                are stored on exit in array E), and */
/* >            b) If UPLO = 'U': factor U in the superdiagonal part of A. */
/* >               If UPLO = 'L': factor L in the subdiagonal part of A. */
/* > */
/* >          2) If WAY = 'R': */
/* > */
/* >          On entry, contains factorization details in format used in */
/* >          ZSYTRF_RK or ZSYTRF_BK: */
/* >            a) ONLY diagonal elements of the symmetric block diagonal */
/* >               matrix D on the diagonal of A, i.e. D(k,k) = A(k,k); */
/* >               (superdiagonal (or subdiagonal) elements of D */
/* >                are stored on exit in array E), and */
/* >            b) If UPLO = 'U': factor U in the superdiagonal part of A. */
/* >               If UPLO = 'L': factor L in the subdiagonal part of A. */
/* > */
/* >          On exit, contains factorization details in format used in */
/* >          ZSYTRF_ROOK: */
/* >            a) all elements of the symmetric block diagonal */
/* >               matrix D on the diagonal of A and on superdiagonal */
/* >               (or subdiagonal) of A, and */
/* >            b) If UPLO = 'U': multipliers used to obtain factor U */
/* >               in the superdiagonal part of A. */
/* >               If UPLO = 'L': multipliers used to obtain factor L */
/* >               in the superdiagonal part of A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] E */
/* > \verbatim */
/* >          E is COMPLEX*16 array, dimension (N) */
/* > */
/* >          1) If WAY ='C': */
/* > */
/* >          On entry, just a workspace. */
/* > */
/* >          On exit, contains the superdiagonal (or subdiagonal) */
/* >          elements of the symmetric block diagonal matrix D */
/* >          with 1-by-1 or 2-by-2 diagonal blocks, where */
/* >          If UPLO = 'U': E(i) = D(i-1,i), i=2:N, E(1) is set to 0; */
/* >          If UPLO = 'L': E(i) = D(i+1,i), i=1:N-1, E(N) is set to 0. */
/* > */
/* >          2) If WAY = 'R': */
/* > */
/* >          On entry, contains the superdiagonal (or subdiagonal) */
/* >          elements of the symmetric block diagonal matrix D */
/* >          with 1-by-1 or 2-by-2 diagonal blocks, where */
/* >          If UPLO = 'U': E(i) = D(i-1,i),i=2:N, E(1) not referenced; */
/* >          If UPLO = 'L': E(i) = D(i+1,i),i=1:N-1, E(N) not referenced. */
/* > */
/* >          On exit, is not changed */
/* > \endverbatim */
/* . */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          On entry, details of the interchanges and the block */
/* >          structure of D as determined: */
/* >          1) by ZSYTRF_ROOK, if WAY ='C'; */
/* >          2) by ZSYTRF_RK (or ZSYTRF_BK), if WAY ='R'. */
/* >          The IPIV format is the same for all these routines. */
/* > */
/* >          On exit, is not changed. */
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

/* > \date November 2017 */

/* > \ingroup complex16SYcomputational */

/* > \par Contributors: */
/*  ================== */
/* > */
/* > \verbatim */
/* > */
/* >  November 2017,  Igor Kozachenko, */
/* >                  Computer Science Division, */
/* >                  University of California, Berkeley */
/* > */
/* > \endverbatim */
/*  ===================================================================== */
void  zsyconvf_rook__(char *uplo, char *way, integer *n, 
	quadcomplex *a, integer *lda, quadcomplex *e, integer *ipiv, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    integer i__, ip, ip2;
    extern logical lsame_(char *, char *);
    logical upper;
    extern void  wswap_(integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *), xerbla_(char *, integer *);
    logical convert;


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */



    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --e;
    --ipiv;

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
	xerbla_("ZSYCONVF_ROOK", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

    if (upper) {

/*        Begin A is UPPER */

	if (convert) {

/*           Convert A (A is upper) */


/*           Convert VALUE */

/*           Assign superdiagonal entries of D to array E and zero out */
/*           corresponding entries in input storage A */

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

/*           Apply permutaions to submatrices of upper part of A */
/*           in factorization order where i decreases from N to 1 */

	    i__ = *n;
	    while(i__ >= 1) {
		if (ipiv[i__] > 0) {

/*                 1-by-1 pivot interchange */

/*                 Swap rows i and IPIV(i) in A(1:i,N-i:N) */

		    ip = ipiv[i__];
		    if (i__ < *n) {
			if (ip != i__) {
			    i__1 = *n - i__;
			    wswap_(&i__1, &a[i__ + (i__ + 1) * a_dim1], lda, &
				    a[ip + (i__ + 1) * a_dim1], lda);
			}
		    }

		} else {

/*                 2-by-2 pivot interchange */

/*                 Swap rows i and IPIV(i) and i-1 and IPIV(i-1) */
/*                 in A(1:i,N-i:N) */

		    ip = -ipiv[i__];
		    ip2 = -ipiv[i__ - 1];
		    if (i__ < *n) {
			if (ip != i__) {
			    i__1 = *n - i__;
			    wswap_(&i__1, &a[i__ + (i__ + 1) * a_dim1], lda, &
				    a[ip + (i__ + 1) * a_dim1], lda);
			}
			if (ip2 != i__ - 1) {
			    i__1 = *n - i__;
			    wswap_(&i__1, &a[i__ - 1 + (i__ + 1) * a_dim1], 
				    lda, &a[ip2 + (i__ + 1) * a_dim1], lda);
			}
		    }
		    --i__;

		}
		--i__;
	    }

	} else {

/*           Revert A (A is upper) */


/*           Revert PERMUTATIONS */

/*           Apply permutaions to submatrices of upper part of A */
/*           in reverse factorization order where i increases from 1 to N */

	    i__ = 1;
	    while(i__ <= *n) {
		if (ipiv[i__] > 0) {

/*                 1-by-1 pivot interchange */

/*                 Swap rows i and IPIV(i) in A(1:i,N-i:N) */

		    ip = ipiv[i__];
		    if (i__ < *n) {
			if (ip != i__) {
			    i__1 = *n - i__;
			    wswap_(&i__1, &a[ip + (i__ + 1) * a_dim1], lda, &
				    a[i__ + (i__ + 1) * a_dim1], lda);
			}
		    }

		} else {

/*                 2-by-2 pivot interchange */

/*                 Swap rows i-1 and IPIV(i-1) and i and IPIV(i) */
/*                 in A(1:i,N-i:N) */

		    ++i__;
		    ip = -ipiv[i__];
		    ip2 = -ipiv[i__ - 1];
		    if (i__ < *n) {
			if (ip2 != i__ - 1) {
			    i__1 = *n - i__;
			    wswap_(&i__1, &a[ip2 + (i__ + 1) * a_dim1], lda, &
				    a[i__ - 1 + (i__ + 1) * a_dim1], lda);
			}
			if (ip != i__) {
			    i__1 = *n - i__;
			    wswap_(&i__1, &a[ip + (i__ + 1) * a_dim1], lda, &
				    a[i__ + (i__ + 1) * a_dim1], lda);
			}
		    }

		}
		++i__;
	    }

/*           Revert VALUE */
/*           Assign superdiagonal entries of D from array E to */
/*           superdiagonal entries of A. */

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

/*        End A is UPPER */

	}

    } else {

/*        Begin A is LOWER */

	if (convert) {

/*           Convert A (A is lower) */


/*           Convert VALUE */
/*           Assign subdiagonal entries of D to array E and zero out */
/*           corresponding entries in input storage A */

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

/*           Apply permutaions to submatrices of lower part of A */
/*           in factorization order where i increases from 1 to N */

	    i__ = 1;
	    while(i__ <= *n) {
		if (ipiv[i__] > 0) {

/*                 1-by-1 pivot interchange */

/*                 Swap rows i and IPIV(i) in A(i:N,1:i-1) */

		    ip = ipiv[i__];
		    if (i__ > 1) {
			if (ip != i__) {
			    i__1 = i__ - 1;
			    wswap_(&i__1, &a[i__ + a_dim1], lda, &a[ip + 
				    a_dim1], lda);
			}
		    }

		} else {

/*                 2-by-2 pivot interchange */

/*                 Swap rows i and IPIV(i) and i+1 and IPIV(i+1) */
/*                 in A(i:N,1:i-1) */

		    ip = -ipiv[i__];
		    ip2 = -ipiv[i__ + 1];
		    if (i__ > 1) {
			if (ip != i__) {
			    i__1 = i__ - 1;
			    wswap_(&i__1, &a[i__ + a_dim1], lda, &a[ip + 
				    a_dim1], lda);
			}
			if (ip2 != i__ + 1) {
			    i__1 = i__ - 1;
			    wswap_(&i__1, &a[i__ + 1 + a_dim1], lda, &a[ip2 + 
				    a_dim1], lda);
			}
		    }
		    ++i__;

		}
		++i__;
	    }

	} else {

/*           Revert A (A is lower) */


/*           Revert PERMUTATIONS */

/*           Apply permutaions to submatrices of lower part of A */
/*           in reverse factorization order where i decreases from N to 1 */

	    i__ = *n;
	    while(i__ >= 1) {
		if (ipiv[i__] > 0) {

/*                 1-by-1 pivot interchange */

/*                 Swap rows i and IPIV(i) in A(i:N,1:i-1) */

		    ip = ipiv[i__];
		    if (i__ > 1) {
			if (ip != i__) {
			    i__1 = i__ - 1;
			    wswap_(&i__1, &a[ip + a_dim1], lda, &a[i__ + 
				    a_dim1], lda);
			}
		    }

		} else {

/*                 2-by-2 pivot interchange */

/*                 Swap rows i+1 and IPIV(i+1) and i and IPIV(i) */
/*                 in A(i:N,1:i-1) */

		    --i__;
		    ip = -ipiv[i__];
		    ip2 = -ipiv[i__ + 1];
		    if (i__ > 1) {
			if (ip2 != i__ + 1) {
			    i__1 = i__ - 1;
			    wswap_(&i__1, &a[ip2 + a_dim1], lda, &a[i__ + 1 + 
				    a_dim1], lda);
			}
			if (ip != i__) {
			    i__1 = i__ - 1;
			    wswap_(&i__1, &a[ip + a_dim1], lda, &a[i__ + 
				    a_dim1], lda);
			}
		    }

		}
		--i__;
	    }

/*           Revert VALUE */
/*           Assign subdiagonal entries of D from array E to */
/*           subgiagonal entries of A. */

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

/*        End A is LOWER */

    }
    return;

/*     End of ZSYCONVF_ROOK */

} /* zsyconvf_rook__ */

