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
static integer c_n1 = -1;
static integer c__3 = 3;
static integer c__2 = 2;

/* > \brief \b ZTZRZF */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZTZRZF + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ztzrzf.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ztzrzf.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ztzrzf.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZTZRZF( M, N, A, LDA, TAU, WORK, LWORK, INFO ) */

/*       INTEGER            INFO, LDA, LWORK, M, N */
/*       COMPLEX*16         A( LDA, * ), TAU( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZTZRZF reduces the M-by-N ( M<=N ) complex upper trapezoidal matrix A */
/* > to upper triangular form by means of unitary transformations. */
/* > */
/* > The upper trapezoidal matrix A is factored as */
/* > */
/* >    A = ( R  0 ) * Z, */
/* > */
/* > where Z is an N-by-N unitary matrix and R is an M-by-M upper */
/* > triangular matrix. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A.  N >= M. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the leading M-by-N upper trapezoidal part of the */
/* >          array A must contain the matrix to be factorized. */
/* >          On exit, the leading M-by-M upper triangular part of A */
/* >          contains the upper triangular matrix R, and elements M+1 to */
/* >          N of the first M rows of A, with the array TAU, represent the */
/* >          unitary matrix Z as a product of M elementary reflectors. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] TAU */
/* > \verbatim */
/* >          TAU is COMPLEX*16 array, dimension (M) */
/* >          The scalar factors of the elementary reflectors. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK.  LWORK >= f2cmax(1,M). */
/* >          For optimum performance LWORK >= M*NB, where NB is */
/* >          the optimal blocksize. */
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
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date April 2012 */

/* > \ingroup complex16OTHERcomputational */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The N-by-N matrix Z can be computed by */
/* > */
/* >     Z =  Z(1)*Z(2)* ... *Z(M) */
/* > */
/* >  where each N-by-N Z(k) is given by */
/* > */
/* >     Z(k) = I - tau(k)*v(k)*v(k)**H */
/* > */
/* >  with v(k) is the kth row vector of the M-by-N matrix */
/* > */
/* >     V = ( I   A(:,M+1:N) ) */
/* > */
/* >  I is the M-by-M identity matrix, A(:,M+1:N) */
/* >  is the output stored in A on exit from DTZRZF, */
/* >  and tau(k) is the kth element of the array TAU. */
/* > */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  wtzrzf_(integer *m, integer *n, quadcomplex *a, 
	integer *lda, quadcomplex *tau, quadcomplex *work, integer *lwork,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    /* Local variables */
    integer i__, m1, ib, nb, ki, kk, mu, nx, iws, nbmin;
    extern void  xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    integer lwkmin, ldwork;
    extern void  wlarzb_(char *, char *, char *, char *, 
	    integer *, integer *, integer *, integer *, quadcomplex *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, integer *,
	     quadcomplex *, integer *);
    integer lwkopt;
    logical lquery;
    extern void  wlarzt_(char *, char *, integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, quadcomplex *, 
	    integer *), wlatrz_(integer *, integer *, integer 
	    *, quadcomplex *, integer *, quadcomplex *, quadcomplex *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     April 2012 */


/*  ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -4;
    }

    if (*info == 0) {
	if (*m == 0 || *m == *n) {
	    lwkopt = 1;
	    lwkmin = 1;
	} else {

/*           Determine the block size. */

	    nb = ilaenv_(&c__1, "ZGERQF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, 
		    (ftnlen)1);
	    lwkopt = *m * nb;
	    lwkmin = f2cmax(1,*m);
	}
	work[1].r = (quadreal) lwkopt, work[1].i = 0.;

	if (*lwork < lwkmin && ! lquery) {
	    *info = -7;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTZRZF", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

    if (*m == 0) {
	return;
    } else if (*m == *n) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    tau[i__2].r = 0., tau[i__2].i = 0.;
/* L10: */
	}
	return;
    }

    nbmin = 2;
    nx = 1;
    iws = *m;
    if (nb > 1 && nb < *m) {

/*        Determine when to cross over from blocked to unblocked code. */

/* Computing MAX */
	i__1 = 0, i__2 = ilaenv_(&c__3, "ZGERQF", " ", m, n, &c_n1, &c_n1, (
		ftnlen)6, (ftnlen)1);
	nx = f2cmax(i__1,i__2);
	if (nx < *m) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *m;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*              Not enough workspace to use optimal NB:  reduce NB and */
/*              determine the minimum value of NB. */

		nb = *lwork / ldwork;
/* Computing MAX */
		i__1 = 2, i__2 = ilaenv_(&c__2, "ZGERQF", " ", m, n, &c_n1, &
			c_n1, (ftnlen)6, (ftnlen)1);
		nbmin = f2cmax(i__1,i__2);
	    }
	}
    }

    if (nb >= nbmin && nb < *m && nx < *m) {

/*        Use blocked code initially. */
/*        The last kk rows are handled by the block method. */

/* Computing MIN */
	i__1 = *m + 1;
	m1 = f2cmin(i__1,*n);
	ki = (*m - nx - 1) / nb * nb;
/* Computing MIN */
	i__1 = *m, i__2 = ki + nb;
	kk = f2cmin(i__1,i__2);

	i__1 = *m - kk + 1;
	i__2 = -nb;
	for (i__ = *m - kk + ki + 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; 
		i__ += i__2) {
/* Computing MIN */
	    i__3 = *m - i__ + 1;
	    ib = f2cmin(i__3,nb);

/*           Compute the TZ factorization of the current block */
/*           A(i:i+ib-1,i:n) */

	    i__3 = *n - i__ + 1;
	    i__4 = *n - *m;
	    wlatrz_(&ib, &i__3, &i__4, &a[i__ + i__ * a_dim1], lda, &tau[i__],
		     &work[1]);
	    if (i__ > 1) {

/*              Form the triangular factor of the block reflector */
/*              H = H(i+ib-1) . . . H(i+1) H(i) */

		i__3 = *n - *m;
		wlarzt_("Backward", "Rowwise", &i__3, &ib, &a[i__ + m1 * 
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(1:i-1,i:n) from the right */

		i__3 = i__ - 1;
		i__4 = *n - i__ + 1;
		i__5 = *n - *m;
		wlarzb_("Right", "No transpose", "Backward", "Rowwise", &i__3,
			 &i__4, &ib, &i__5, &a[i__ + m1 * a_dim1], lda, &work[
			1], &ldwork, &a[i__ * a_dim1 + 1], lda, &work[ib + 1],
			 &ldwork)
			;
	    }
/* L20: */
	}
	mu = i__ + nb - 1;
    } else {
	mu = *m;
    }

/*     Use unblocked code to factor the last or only block */

    if (mu > 0) {
	i__2 = *n - *m;
	wlatrz_(&mu, n, &i__2, &a[a_offset], lda, &tau[1], &work[1]);
    }

    work[1].r = (quadreal) lwkopt, work[1].i = 0.;

    return;

/*     End of ZTZRZF */

} /* wtzrzf_ */

