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

/* Table of constant values */

static integer c__1 = 1;

/* > \brief \b DSYTF2_ROOK computes the factorization of a doublereal symmetric indefinite matrix using the bounded 
Bunch-Kaufman ("rook") diagonal pivoting method (unblocked algorithm). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DSYTF2_ROOK + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsytf2_
rook.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsytf2_
rook.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsytf2_
rook.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DSYTF2_ROOK( UPLO, N, A, LDA, IPIV, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, N */
/*       INTEGER            IPIV( * ) */
/*       DOUBLE PRECISION   A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DSYTF2_ROOK computes the factorization of a doublereal symmetric matrix A */
/* > using the bounded Bunch-Kaufman ("rook") diagonal pivoting method: */
/* > */
/* >    A = U*D*U**T  or  A = L*D*L**T */
/* > */
/* > where U (or L) is a product of permutation and unit upper (lower) */
/* > triangular matrices, U**T is the transpose of U, and D is symmetric and */
/* > block diagonal with 1-by-1 and 2-by-2 diagonal blocks. */
/* > */
/* > This is the unblocked version of the algorithm, calling Level 2 BLAS. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the upper or lower triangular part of the */
/* >          symmetric matrix A is stored: */
/* >          = 'U':  Upper triangular */
/* >          = 'L':  Lower triangular */
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
/* >          On entry, the symmetric matrix A.  If UPLO = 'U', the leading */
/* >          n-by-n upper triangular part of A contains the upper */
/* >          triangular part of the matrix A, and the strictly lower */
/* >          triangular part of A is not referenced.  If UPLO = 'L', the */
/* >          leading n-by-n lower triangular part of A contains the lower */
/* >          triangular part of the matrix A, and the strictly upper */
/* >          triangular part of A is not referenced. */
/* > */
/* >          On exit, the block diagonal matrix D and the multipliers used */
/* >          to obtain the factor U or L (see below for further details). */
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
/* >          Details of the interchanges and the block structure of D. */
/* > */
/* >          If UPLO = 'U': */
/* >             If IPIV(k) > 0, then rows and columns k and IPIV(k) */
/* >             were interchanged and D(k,k) is a 1-by-1 diagonal block. */
/* > */
/* >             If IPIV(k) < 0 and IPIV(k-1) < 0, then rows and */
/* >             columns k and -IPIV(k) were interchanged and rows and */
/* >             columns k-1 and -IPIV(k-1) were inerchaged, */
/* >             D(k-1:k,k-1:k) is a 2-by-2 diagonal block. */
/* > */
/* >          If UPLO = 'L': */
/* >             If IPIV(k) > 0, then rows and columns k and IPIV(k) */
/* >             were interchanged and D(k,k) is a 1-by-1 diagonal block. */
/* > */
/* >             If IPIV(k) < 0 and IPIV(k+1) < 0, then rows and */
/* >             columns k and -IPIV(k) were interchanged and rows and */
/* >             columns k+1 and -IPIV(k+1) were inerchaged, */
/* >             D(k:k+1,k:k+1) is a 2-by-2 diagonal block. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -k, the k-th argument had an illegal value */
/* >          > 0: if INFO = k, D(k,k) is exactly zero.  The factorization */
/* >               has been completed, but the block diagonal matrix D is */
/* >               exactly singular, and division by zero will occur if it */
/* >               is used to solve a system of equations. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2013 */

/* > \ingroup doubleSYcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  If UPLO = 'U', then A = U*D*U**T, where */
/* >     U = P(n)*U(n)* ... *P(k)U(k)* ..., */
/* >  i.e., U is a product of terms P(k)*U(k), where k decreases from n to */
/* >  1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1 */
/* >  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as */
/* >  defined by IPIV(k), and U(k) is a unit upper triangular matrix, such */
/* >  that if the diagonal block D(k) is of order s (s = 1 or 2), then */
/* > */
/* >             (   I    v    0   )   k-s */
/* >     U(k) =  (   0    I    0   )   s */
/* >             (   0    0    I   )   n-k */
/* >                k-s   s   n-k */
/* > */
/* >  If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k). */
/* >  If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k), */
/* >  and A(k,k), and v overwrites A(1:k-2,k-1:k). */
/* > */
/* >  If UPLO = 'L', then A = L*D*L**T, where */
/* >     L = P(1)*L(1)* ... *P(k)*L(k)* ..., */
/* >  i.e., L is a product of terms P(k)*L(k), where k increases from 1 to */
/* >  n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1 */
/* >  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as */
/* >  defined by IPIV(k), and L(k) is a unit lower triangular matrix, such */
/* >  that if the diagonal block D(k) is of order s (s = 1 or 2), then */
/* > */
/* >             (   I    0     0   )  k-1 */
/* >     L(k) =  (   0    I     0   )  s */
/* >             (   0    v     I   )  n-k-s+1 */
/* >                k-1   s  n-k-s+1 */
/* > */
/* >  If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k). */
/* >  If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k), */
/* >  and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1). */
/* > \endverbatim */

/* > \par Contributors: */
/*  ================== */
/* > */
/* > \verbatim */
/* > */
/* >  November 2013,     Igor Kozachenko, */
/* >                  Computer Science Division, */
/* >                  University of California, Berkeley */
/* > */
/* >  September 2007, Sven Hammarling, Nicholas J. Higham, Craig Lucas, */
/* >                  School of Mathematics, */
/* >                  University of Manchester */
/* > */
/* >  01-01-96 - Based on modifications by */
/* >    J. Lewis, Boeing Computer Services Company */
/* >    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville abd , USA */
/* > \endverbatim */

/*  ===================================================================== */
void  dsytf2_rook__(char *uplo, integer *n, quadreal *a, 
	integer *lda, integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    quadreal d__1;

    /* Local variables */
    integer i__, j, k, p;
    quadreal t, d11, d12, d21, d22;
    integer ii, kk, kp;
    quadreal wk, wkm1, wkp1;
    logical done;
    integer imax, jmax;
    extern void  qsyr_(char *, integer *, quadreal *, 
	    quadreal *, integer *, quadreal *, integer *);
    quadreal alpha;
    extern void  qscal_(integer *, quadreal *, quadreal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    quadreal dtemp, sfmin;
    integer itemp;
    extern void  qswap_(integer *, quadreal *, integer *, 
	    quadreal *, integer *);
    integer kstep;
    logical upper;
    extern quadreal qlamch_(char *);
    quadreal absakk;
    extern integer iqamax_(integer *, quadreal *, integer *);
    extern void  xerbla_(char *, integer *);
    quadreal colmax, rowmax;


/*  -- LAPACK computational routine (version 3.5.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2013 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;

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
	xerbla_("DSYTF2_ROOK", &i__1);
	return;
    }

/*     Initialize ALPHA for use in choosing pivot block size. */

    alpha = (M(sqrt)(17.) + 1.) / 8.;

/*     Compute machine safe minimum */

    sfmin = qlamch_("S");

    if (upper) {

/*        Factorize A as U*D*U**T using the upper triangle of A */

/*        K is the main loop index, decreasing from N to 1 in steps of */
/*        1 or 2 */

	k = *n;
L10:

/*        If K < 1, exit from loop */

	if (k < 1) {
	    goto L70;
	}
	kstep = 1;
	p = k;

/*        Determine rows and columns to be interchanged and whether */
/*        a 1-by-1 or 2-by-2 pivot block will be used */

	absakk = (d__1 = a[k + k * a_dim1], abs(d__1));

/*        IMAX is the row-index of the largest off-diagonal element in */
/*        column K, and COLMAX is its absolute value. */
/*        Determine both COLMAX and IMAX. */

	if (k > 1) {
	    i__1 = k - 1;
	    imax = iqamax_(&i__1, &a[k * a_dim1 + 1], &c__1);
	    colmax = (d__1 = a[imax + k * a_dim1], abs(d__1));
	} else {
	    colmax = 0.;
	}

	if (f2cmax(absakk,colmax) == 0.) {

/*           Column K is zero or underflow: set INFO and continue */

	    if (*info == 0) {
		*info = k;
	    }
	    kp = k;
	} else {

/*           Test for interchange */

/*           Equivalent to testing for (used to handle NaN and Inf) */
/*           ABSAKK.GE.ALPHA*COLMAX */

	    if (! (absakk < alpha * colmax)) {

/*              no interchange, */
/*              use 1-by-1 pivot block */

		kp = k;
	    } else {

		done = FALSE_;

/*              Loop until pivot found */

L12:

/*                 Begin pivot search loop body */

/*                 JMAX is the column-index of the largest off-diagonal */
/*                 element in row IMAX, and ROWMAX is its absolute value. */
/*                 Determine both ROWMAX and JMAX. */

		if (imax != k) {
		    i__1 = k - imax;
		    jmax = imax + iqamax_(&i__1, &a[imax + (imax + 1) * 
			    a_dim1], lda);
		    rowmax = (d__1 = a[imax + jmax * a_dim1], abs(d__1));
		} else {
		    rowmax = 0.;
		}

		if (imax > 1) {
		    i__1 = imax - 1;
		    itemp = iqamax_(&i__1, &a[imax * a_dim1 + 1], &c__1);
		    dtemp = (d__1 = a[itemp + imax * a_dim1], abs(d__1));
		    if (dtemp > rowmax) {
			rowmax = dtemp;
			jmax = itemp;
		    }
		}

/*                 Equivalent to testing for (used to handle NaN and Inf) */
/*                 ABS( A( IMAX, IMAX ) ).GE.ALPHA*ROWMAX */

		if (! ((d__1 = a[imax + imax * a_dim1], abs(d__1)) < alpha * 
			rowmax)) {

/*                    interchange rows and columns K and IMAX, */
/*                    use 1-by-1 pivot block */

		    kp = imax;
		    done = TRUE_;

/*                 Equivalent to testing for ROWMAX .EQ. COLMAX, */
/*                 used to handle NaN and Inf */

		} else if (p == jmax || rowmax <= colmax) {

/*                    interchange rows and columns K+1 and IMAX, */
/*                    use 2-by-2 pivot block */

		    kp = imax;
		    kstep = 2;
		    done = TRUE_;
		} else {

/*                    Pivot NOT found, set variables and repeat */

		    p = imax;
		    colmax = rowmax;
		    imax = jmax;
		}

/*                 End pivot search loop body */

		if (! done) {
		    goto L12;
		}

	    }

/*           Swap TWO rows and TWO columns */

/*           First swap */

	    if (kstep == 2 && p != k) {

/*              Interchange rows and column K and P in the leading */
/*              submatrix A(1:k,1:k) if we have a 2-by-2 pivot */

		if (p > 1) {
		    i__1 = p - 1;
		    qswap_(&i__1, &a[k * a_dim1 + 1], &c__1, &a[p * a_dim1 + 
			    1], &c__1);
		}
		if (p < k - 1) {
		    i__1 = k - p - 1;
		    qswap_(&i__1, &a[p + 1 + k * a_dim1], &c__1, &a[p + (p + 
			    1) * a_dim1], lda);
		}
		t = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[p + p * a_dim1];
		a[p + p * a_dim1] = t;
	    }

/*           Second swap */

	    kk = k - kstep + 1;
	    if (kp != kk) {

/*              Interchange rows and columns KK and KP in the leading */
/*              submatrix A(1:k,1:k) */

		if (kp > 1) {
		    i__1 = kp - 1;
		    qswap_(&i__1, &a[kk * a_dim1 + 1], &c__1, &a[kp * a_dim1 
			    + 1], &c__1);
		}
		if (kk > 1 && kp < kk - 1) {
		    i__1 = kk - kp - 1;
		    qswap_(&i__1, &a[kp + 1 + kk * a_dim1], &c__1, &a[kp + (
			    kp + 1) * a_dim1], lda);
		}
		t = a[kk + kk * a_dim1];
		a[kk + kk * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = t;
		if (kstep == 2) {
		    t = a[k - 1 + k * a_dim1];
		    a[k - 1 + k * a_dim1] = a[kp + k * a_dim1];
		    a[kp + k * a_dim1] = t;
		}
	    }

/*           Update the leading submatrix */

	    if (kstep == 1) {

/*              1-by-1 pivot block D(k): column k now holds */

/*              W(k) = U(k)*D(k) */

/*              where U(k) is the k-th column of U */

		if (k > 1) {

/*                 Perform a rank-1 update of A(1:k-1,1:k-1) and */
/*                 store U(k) in column k */

		    if ((d__1 = a[k + k * a_dim1], abs(d__1)) >= sfmin) {

/*                    Perform a rank-1 update of A(1:k-1,1:k-1) as */
/*                    A := A - U(k)*D(k)*U(k)**T */
/*                       = A - W(k)*1/D(k)*W(k)**T */

			d11 = 1. / a[k + k * a_dim1];
			i__1 = k - 1;
			d__1 = -d11;
			qsyr_(uplo, &i__1, &d__1, &a[k * a_dim1 + 1], &c__1, &
				a[a_offset], lda);

/*                    Store U(k) in column k */

			i__1 = k - 1;
			qscal_(&i__1, &d11, &a[k * a_dim1 + 1], &c__1);
		    } else {

/*                    Store L(k) in column K */

			d11 = a[k + k * a_dim1];
			i__1 = k - 1;
			for (ii = 1; ii <= i__1; ++ii) {
			    a[ii + k * a_dim1] /= d11;
/* L16: */
			}

/*                    Perform a rank-1 update of A(k+1:n,k+1:n) as */
/*                    A := A - U(k)*D(k)*U(k)**T */
/*                       = A - W(k)*(1/D(k))*W(k)**T */
/*                       = A - (W(k)/D(k))*(D(k))*(W(k)/D(K))**T */

			i__1 = k - 1;
			d__1 = -d11;
			qsyr_(uplo, &i__1, &d__1, &a[k * a_dim1 + 1], &c__1, &
				a[a_offset], lda);
		    }
		}

	    } else {

/*              2-by-2 pivot block D(k): columns k and k-1 now hold */

/*              ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k) */

/*              where U(k) and U(k-1) are the k-th and (k-1)-th columns */
/*              of U */

/*              Perform a rank-2 update of A(1:k-2,1:k-2) as */

/*              A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )**T */
/*                 = A - ( ( A(k-1)A(k) )*inv(D(k)) ) * ( A(k-1)A(k) )**T */

/*              and store L(k) and L(k+1) in columns k and k+1 */

		if (k > 2) {

		    d12 = a[k - 1 + k * a_dim1];
		    d22 = a[k - 1 + (k - 1) * a_dim1] / d12;
		    d11 = a[k + k * a_dim1] / d12;
		    t = 1. / (d11 * d22 - 1.);

		    for (j = k - 2; j >= 1; --j) {

			wkm1 = t * (d11 * a[j + (k - 1) * a_dim1] - a[j + k * 
				a_dim1]);
			wk = t * (d22 * a[j + k * a_dim1] - a[j + (k - 1) * 
				a_dim1]);

			for (i__ = j; i__ >= 1; --i__) {
			    a[i__ + j * a_dim1] = a[i__ + j * a_dim1] - a[i__ 
				    + k * a_dim1] / d12 * wk - a[i__ + (k - 1)
				     * a_dim1] / d12 * wkm1;
/* L20: */
			}

/*                    Store U(k) and U(k-1) in cols k and k-1 for row J */

			a[j + k * a_dim1] = wk / d12;
			a[j + (k - 1) * a_dim1] = wkm1 / d12;

/* L30: */
		    }

		}

	    }
	}

/*        Store details of the interchanges in IPIV */

	if (kstep == 1) {
	    ipiv[k] = kp;
	} else {
	    ipiv[k] = -p;
	    ipiv[k - 1] = -kp;
	}

/*        Decrease K and return to the start of the main loop */

	k -= kstep;
	goto L10;

    } else {

/*        Factorize A as L*D*L**T using the lower triangle of A */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2 */

	k = 1;
L40:

/*        If K > N, exit from loop */

	if (k > *n) {
	    goto L70;
	}
	kstep = 1;
	p = k;

/*        Determine rows and columns to be interchanged and whether */
/*        a 1-by-1 or 2-by-2 pivot block will be used */

	absakk = (d__1 = a[k + k * a_dim1], abs(d__1));

/*        IMAX is the row-index of the largest off-diagonal element in */
/*        column K, and COLMAX is its absolute value. */
/*        Determine both COLMAX and IMAX. */

	if (k < *n) {
	    i__1 = *n - k;
	    imax = k + iqamax_(&i__1, &a[k + 1 + k * a_dim1], &c__1);
	    colmax = (d__1 = a[imax + k * a_dim1], abs(d__1));
	} else {
	    colmax = 0.;
	}

	if (f2cmax(absakk,colmax) == 0.) {

/*           Column K is zero or underflow: set INFO and continue */

	    if (*info == 0) {
		*info = k;
	    }
	    kp = k;
	} else {

/*           Test for interchange */

/*           Equivalent to testing for (used to handle NaN and Inf) */
/*           ABSAKK.GE.ALPHA*COLMAX */

	    if (! (absakk < alpha * colmax)) {

/*              no interchange, use 1-by-1 pivot block */

		kp = k;
	    } else {

		done = FALSE_;

/*              Loop until pivot found */

L42:

/*                 Begin pivot search loop body */

/*                 JMAX is the column-index of the largest off-diagonal */
/*                 element in row IMAX, and ROWMAX is its absolute value. */
/*                 Determine both ROWMAX and JMAX. */

		if (imax != k) {
		    i__1 = imax - k;
		    jmax = k - 1 + iqamax_(&i__1, &a[imax + k * a_dim1], lda);
		    rowmax = (d__1 = a[imax + jmax * a_dim1], abs(d__1));
		} else {
		    rowmax = 0.;
		}

		if (imax < *n) {
		    i__1 = *n - imax;
		    itemp = imax + iqamax_(&i__1, &a[imax + 1 + imax * a_dim1]
			    , &c__1);
		    dtemp = (d__1 = a[itemp + imax * a_dim1], abs(d__1));
		    if (dtemp > rowmax) {
			rowmax = dtemp;
			jmax = itemp;
		    }
		}

/*                 Equivalent to testing for (used to handle NaN and Inf) */
/*                 ABS( A( IMAX, IMAX ) ).GE.ALPHA*ROWMAX */

		if (! ((d__1 = a[imax + imax * a_dim1], abs(d__1)) < alpha * 
			rowmax)) {

/*                    interchange rows and columns K and IMAX, */
/*                    use 1-by-1 pivot block */

		    kp = imax;
		    done = TRUE_;

/*                 Equivalent to testing for ROWMAX .EQ. COLMAX, */
/*                 used to handle NaN and Inf */

		} else if (p == jmax || rowmax <= colmax) {

/*                    interchange rows and columns K+1 and IMAX, */
/*                    use 2-by-2 pivot block */

		    kp = imax;
		    kstep = 2;
		    done = TRUE_;
		} else {

/*                    Pivot NOT found, set variables and repeat */

		    p = imax;
		    colmax = rowmax;
		    imax = jmax;
		}

/*                 End pivot search loop body */

		if (! done) {
		    goto L42;
		}

	    }

/*           Swap TWO rows and TWO columns */

/*           First swap */

	    if (kstep == 2 && p != k) {

/*              Interchange rows and column K and P in the trailing */
/*              submatrix A(k:n,k:n) if we have a 2-by-2 pivot */

		if (p < *n) {
		    i__1 = *n - p;
		    qswap_(&i__1, &a[p + 1 + k * a_dim1], &c__1, &a[p + 1 + p 
			    * a_dim1], &c__1);
		}
		if (p > k + 1) {
		    i__1 = p - k - 1;
		    qswap_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &a[p + (k + 
			    1) * a_dim1], lda);
		}
		t = a[k + k * a_dim1];
		a[k + k * a_dim1] = a[p + p * a_dim1];
		a[p + p * a_dim1] = t;
	    }

/*           Second swap */

	    kk = k + kstep - 1;
	    if (kp != kk) {

/*              Interchange rows and columns KK and KP in the trailing */
/*              submatrix A(k:n,k:n) */

		if (kp < *n) {
		    i__1 = *n - kp;
		    qswap_(&i__1, &a[kp + 1 + kk * a_dim1], &c__1, &a[kp + 1 
			    + kp * a_dim1], &c__1);
		}
		if (kk < *n && kp > kk + 1) {
		    i__1 = kp - kk - 1;
		    qswap_(&i__1, &a[kk + 1 + kk * a_dim1], &c__1, &a[kp + (
			    kk + 1) * a_dim1], lda);
		}
		t = a[kk + kk * a_dim1];
		a[kk + kk * a_dim1] = a[kp + kp * a_dim1];
		a[kp + kp * a_dim1] = t;
		if (kstep == 2) {
		    t = a[k + 1 + k * a_dim1];
		    a[k + 1 + k * a_dim1] = a[kp + k * a_dim1];
		    a[kp + k * a_dim1] = t;
		}
	    }

/*           Update the trailing submatrix */

	    if (kstep == 1) {

/*              1-by-1 pivot block D(k): column k now holds */

/*              W(k) = L(k)*D(k) */

/*              where L(k) is the k-th column of L */

		if (k < *n) {

/*              Perform a rank-1 update of A(k+1:n,k+1:n) and */
/*              store L(k) in column k */

		    if ((d__1 = a[k + k * a_dim1], abs(d__1)) >= sfmin) {

/*                    Perform a rank-1 update of A(k+1:n,k+1:n) as */
/*                    A := A - L(k)*D(k)*L(k)**T */
/*                       = A - W(k)*(1/D(k))*W(k)**T */

			d11 = 1. / a[k + k * a_dim1];
			i__1 = *n - k;
			d__1 = -d11;
			qsyr_(uplo, &i__1, &d__1, &a[k + 1 + k * a_dim1], &
				c__1, &a[k + 1 + (k + 1) * a_dim1], lda);

/*                    Store L(k) in column k */

			i__1 = *n - k;
			qscal_(&i__1, &d11, &a[k + 1 + k * a_dim1], &c__1);
		    } else {

/*                    Store L(k) in column k */

			d11 = a[k + k * a_dim1];
			i__1 = *n;
			for (ii = k + 1; ii <= i__1; ++ii) {
			    a[ii + k * a_dim1] /= d11;
/* L46: */
			}

/*                    Perform a rank-1 update of A(k+1:n,k+1:n) as */
/*                    A := A - L(k)*D(k)*L(k)**T */
/*                       = A - W(k)*(1/D(k))*W(k)**T */
/*                       = A - (W(k)/D(k))*(D(k))*(W(k)/D(K))**T */

			i__1 = *n - k;
			d__1 = -d11;
			qsyr_(uplo, &i__1, &d__1, &a[k + 1 + k * a_dim1], &
				c__1, &a[k + 1 + (k + 1) * a_dim1], lda);
		    }
		}

	    } else {

/*              2-by-2 pivot block D(k): columns k and k+1 now hold */

/*              ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k) */

/*              where L(k) and L(k+1) are the k-th and (k+1)-th columns */
/*              of L */


/*              Perform a rank-2 update of A(k+2:n,k+2:n) as */

/*              A := A - ( L(k) L(k+1) ) * D(k) * ( L(k) L(k+1) )**T */
/*                 = A - ( ( A(k)A(k+1) )*inv(D(k) ) * ( A(k)A(k+1) )**T */

/*              and store L(k) and L(k+1) in columns k and k+1 */

		if (k < *n - 1) {

		    d21 = a[k + 1 + k * a_dim1];
		    d11 = a[k + 1 + (k + 1) * a_dim1] / d21;
		    d22 = a[k + k * a_dim1] / d21;
		    t = 1. / (d11 * d22 - 1.);

		    i__1 = *n;
		    for (j = k + 2; j <= i__1; ++j) {

/*                    Compute  D21 * ( W(k)W(k+1) ) * inv(D(k)) for row J */

			wk = t * (d11 * a[j + k * a_dim1] - a[j + (k + 1) * 
				a_dim1]);
			wkp1 = t * (d22 * a[j + (k + 1) * a_dim1] - a[j + k * 
				a_dim1]);

/*                    Perform a rank-2 update of A(k+2:n,k+2:n) */

			i__2 = *n;
			for (i__ = j; i__ <= i__2; ++i__) {
			    a[i__ + j * a_dim1] = a[i__ + j * a_dim1] - a[i__ 
				    + k * a_dim1] / d21 * wk - a[i__ + (k + 1)
				     * a_dim1] / d21 * wkp1;
/* L50: */
			}

/*                    Store L(k) and L(k+1) in cols k and k+1 for row J */

			a[j + k * a_dim1] = wk / d21;
			a[j + (k + 1) * a_dim1] = wkp1 / d21;

/* L60: */
		    }

		}

	    }
	}

/*        Store details of the interchanges in IPIV */

	if (kstep == 1) {
	    ipiv[k] = kp;
	} else {
	    ipiv[k] = -p;
	    ipiv[k + 1] = -kp;
	}

/*        Increase K and return to the start of the main loop */

	k += kstep;
	goto L40;

    }

L70:

    return;

/*     End of DSYTF2_ROOK */

} /* dsytf2_rook__ */

