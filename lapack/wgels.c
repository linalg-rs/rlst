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

static quadcomplex c_b1 = {0.,0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static integer c__0 = 0;

/* > \brief <b> ZGELS solves overdetermined or underdetermined systems for GE matrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGELS + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgels.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgels.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgels.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGELS( TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK, */
/*                         INFO ) */

/*       CHARACTER          TRANS */
/*       INTEGER            INFO, LDA, LDB, LWORK, M, N, NRHS */
/*       COMPLEX*16         A( LDA, * ), B( LDB, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGELS solves overdetermined or underdetermined complex linear systems */
/* > involving an M-by-N matrix A, or its conjugate-transpose, using a QR */
/* > or LQ factorization of A.  It is assumed that A has full rank. */
/* > */
/* > The following options are provided: */
/* > */
/* > 1. If TRANS = 'N' and m >= n:  find the least squares solution of */
/* >    an overdetermined system, i.e., solve the least squares problem */
/* >                 minimize || B - A*X ||. */
/* > */
/* > 2. If TRANS = 'N' and m < n:  find the minimum norm solution of */
/* >    an underdetermined system A * X = B. */
/* > */
/* > 3. If TRANS = 'C' and m >= n:  find the minimum norm solution of */
/* >    an underdetermined system A**H * X = B. */
/* > */
/* > 4. If TRANS = 'C' and m < n:  find the least squares solution of */
/* >    an overdetermined system, i.e., solve the least squares problem */
/* >                 minimize || B - A**H * X ||. */
/* > */
/* > Several right hand side vectors b and solution vectors x can be */
/* > handled in a single call; they are stored as the columns of the */
/* > M-by-NRHS right hand side matrix B and the N-by-NRHS solution */
/* > matrix X. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] TRANS */
/* > \verbatim */
/* >          TRANS is CHARACTER*1 */
/* >          = 'N': the linear system involves A; */
/* >          = 'C': the linear system involves A**H. */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right hand sides, i.e., the number of */
/* >          columns of the matrices B and X. NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the M-by-N matrix A. */
/* >            if M >= N, A is overwritten by details of its QR */
/* >                       factorization as returned by ZGEQRF; */
/* >            if M <  N, A is overwritten by details of its LQ */
/* >                       factorization as returned by ZGELQF. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,NRHS) */
/* >          On entry, the matrix B of right hand side vectors, stored */
/* >          columnwise; B is M-by-NRHS if TRANS = 'N', or N-by-NRHS */
/* >          if TRANS = 'C'. */
/* >          On exit, if INFO = 0, B is overwritten by the solution */
/* >          vectors, stored columnwise: */
/* >          if TRANS = 'N' and m >= n, rows 1 to n of B contain the least */
/* >          squares solution vectors; the residual sum of squares for the */
/* >          solution in each column is given by the sum of squares of the */
/* >          modulus of elements N+1 to M in that column; */
/* >          if TRANS = 'N' and m < n, rows 1 to N of B contain the */
/* >          minimum norm solution vectors; */
/* >          if TRANS = 'C' and m >= n, rows 1 to M of B contain the */
/* >          minimum norm solution vectors; */
/* >          if TRANS = 'C' and m < n, rows 1 to M of B contain the */
/* >          least squares solution vectors; the residual sum of squares */
/* >          for the solution in each column is given by the sum of */
/* >          squares of the modulus of elements M+1 to N in that column. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. LDB >= MAX(1,M,N). */
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
/* >          The dimension of the array WORK. */
/* >          LWORK >= f2cmax( 1, MN + f2cmax( MN, NRHS ) ). */
/* >          For optimal performance, */
/* >          LWORK >= f2cmax( 1, MN + f2cmax( MN, NRHS )*NB ). */
/* >          where MN = f2cmin(M,N) and NB is the optimum block size. */
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
/* >          > 0:  if INFO =  i, the i-th diagonal element of the */
/* >                triangular factor of A is zero, so that A does not have */
/* >                full rank; the least squares solution could not be */
/* >                computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16GEsolve */

/*  ===================================================================== */
void  wgels_(char *trans, integer *m, integer *n, integer *
	nrhs, quadcomplex *a, integer *lda, quadcomplex *b, integer *ldb, 
	quadcomplex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    quadreal d__1;

    /* Local variables */
    integer i__, j, nb, mn;
    quadreal anrm, bnrm;
    integer brow;
    logical tpsd;
    integer iascl, ibscl;
    extern logical lsame_(char *, char *);
    integer wsize;
    quadreal rwork[1];
    extern void  qlabad_(quadreal *, quadreal *);
    extern quadreal qlamch_(char *);
    extern void  xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    integer scllen;
    quadreal bignum;
    extern quadreal wlange_(char *, integer *, integer *, quadcomplex *, 
	    integer *, quadreal *);
    extern void  wgelqf_(integer *, integer *, quadcomplex *,
	     integer *, quadcomplex *, quadcomplex *, integer *, integer *
	    ), wlascl_(char *, integer *, integer *, quadreal *, quadreal 
	    *, integer *, integer *, quadcomplex *, integer *, integer *), wgeqrf_(integer *, integer *, quadcomplex *, integer *,
	     quadcomplex *, quadcomplex *, integer *, integer *), wlaset_(
	    char *, integer *, integer *, quadcomplex *, quadcomplex *, 
	    quadcomplex *, integer *);
    quadreal smlnum;
    logical lquery;
    extern void  wunmlq_(char *, char *, integer *, integer *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, integer *), wunmqr_(char *, char *, integer *, integer *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, integer *), wtrtrs_(char *, char *, char *, integer *, 
	    integer *, quadcomplex *, integer *, quadcomplex *, integer *,
	     integer *);


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;

    /* Function Body */
    *info = 0;
    mn = f2cmin(*m,*n);
    lquery = *lwork == -1;
    if (! (lsame_(trans, "N") || lsame_(trans, "C"))) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 0) {
	*info = -4;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -6;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = f2cmax(1,*m);
	if (*ldb < f2cmax(i__1,*n)) {
	    *info = -8;
	} else /* if(complicated condition) */ {
/* Computing MAX */
	    i__1 = 1, i__2 = mn + f2cmax(mn,*nrhs);
	    if (*lwork < f2cmax(i__1,i__2) && ! lquery) {
		*info = -10;
	    }
	}
    }

/*     Figure out optimal block size */

    if (*info == 0 || *info == -10) {

	tpsd = TRUE_;
	if (lsame_(trans, "N")) {
	    tpsd = FALSE_;
	}

	if (*m >= *n) {
	    nb = ilaenv_(&c__1, "ZGEQRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, 
		    (ftnlen)1);
	    if (tpsd) {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "ZUNMQR", "LN", m, nrhs, n, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = f2cmax(i__1,i__2);
	    } else {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "ZUNMQR", "LC", m, nrhs, n, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = f2cmax(i__1,i__2);
	    }
	} else {
	    nb = ilaenv_(&c__1, "ZGELQF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, 
		    (ftnlen)1);
	    if (tpsd) {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "ZUNMLQ", "LC", n, nrhs, m, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = f2cmax(i__1,i__2);
	    } else {
/* Computing MAX */
		i__1 = nb, i__2 = ilaenv_(&c__1, "ZUNMLQ", "LN", n, nrhs, m, &
			c_n1, (ftnlen)6, (ftnlen)2);
		nb = f2cmax(i__1,i__2);
	    }
	}

/* Computing MAX */
	i__1 = 1, i__2 = mn + f2cmax(mn,*nrhs) * nb;
	wsize = f2cmax(i__1,i__2);
	d__1 = (quadreal) wsize;
	work[1].r = d__1, work[1].i = 0.;

    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGELS ", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

/* Computing MIN */
    i__1 = f2cmin(*m,*n);
    if (f2cmin(i__1,*nrhs) == 0) {
	i__1 = f2cmax(*m,*n);
	wlaset_("Full", &i__1, nrhs, &c_b1, &c_b1, &b[b_offset], ldb);
	return;
    }

/*     Get machine parameters */

    smlnum = qlamch_("S") / qlamch_("P");
    bignum = 1. / smlnum;
    qlabad_(&smlnum, &bignum);

/*     Scale A, B if f2cmax element outside range [SMLNUM,BIGNUM] */

    anrm = wlange_("M", m, n, &a[a_offset], lda, rwork);
    iascl = 0;
    if (anrm > 0. && anrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	wlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, 
		info);
	iascl = 1;
    } else if (anrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	wlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, 
		info);
	iascl = 2;
    } else if (anrm == 0.) {

/*        Matrix all zero. Return zero solution. */

	i__1 = f2cmax(*m,*n);
	wlaset_("F", &i__1, nrhs, &c_b1, &c_b1, &b[b_offset], ldb);
	goto L50;
    }

    brow = *m;
    if (tpsd) {
	brow = *n;
    }
    bnrm = wlange_("M", &brow, nrhs, &b[b_offset], ldb, rwork);
    ibscl = 0;
    if (bnrm > 0. && bnrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	wlascl_("G", &c__0, &c__0, &bnrm, &smlnum, &brow, nrhs, &b[b_offset], 
		ldb, info);
	ibscl = 1;
    } else if (bnrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	wlascl_("G", &c__0, &c__0, &bnrm, &bignum, &brow, nrhs, &b[b_offset], 
		ldb, info);
	ibscl = 2;
    }

    if (*m >= *n) {

/*        compute QR factorization of A */

	i__1 = *lwork - mn;
	wgeqrf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
		;

/*        workspace at least N, optimally N*NB */

	if (! tpsd) {

/*           Least-Squares Problem f2cmin || A * X - B || */

/*           B(1:M,1:NRHS) := Q**H * B(1:M,1:NRHS) */

	    i__1 = *lwork - mn;
	    wunmqr_("Left", "Conjugate transpose", m, nrhs, n, &a[a_offset], 
		    lda, &work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, 
		    info);

/*           workspace at least NRHS, optimally NRHS*NB */

/*           B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS) */

	    wtrtrs_("Upper", "No transpose", "Non-unit", n, nrhs, &a[a_offset]
		    , lda, &b[b_offset], ldb, info);

	    if (*info > 0) {
		return;
	    }

	    scllen = *n;

	} else {

/*           Underdetermined system of equations A**T * X = B */

/*           B(1:N,1:NRHS) := inv(R**H) * B(1:N,1:NRHS) */

	    wtrtrs_("Upper", "Conjugate transpose", "Non-unit", n, nrhs, &a[
		    a_offset], lda, &b[b_offset], ldb, info);

	    if (*info > 0) {
		return;
	    }

/*           B(N+1:M,1:NRHS) = ZERO */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = *n + 1; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    b[i__3].r = 0., b[i__3].i = 0.;
/* L10: */
		}
/* L20: */
	    }

/*           B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS) */

	    i__1 = *lwork - mn;
	    wunmqr_("Left", "No transpose", m, nrhs, n, &a[a_offset], lda, &
		    work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);

/*           workspace at least NRHS, optimally NRHS*NB */

	    scllen = *m;

	}

    } else {

/*        Compute LQ factorization of A */

	i__1 = *lwork - mn;
	wgelqf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
		;

/*        workspace at least M, optimally M*NB. */

	if (! tpsd) {

/*           underdetermined system of equations A * X = B */

/*           B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS) */

	    wtrtrs_("Lower", "No transpose", "Non-unit", m, nrhs, &a[a_offset]
		    , lda, &b[b_offset], ldb, info);

	    if (*info > 0) {
		return;
	    }

/*           B(M+1:N,1:NRHS) = 0 */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = *m + 1; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    b[i__3].r = 0., b[i__3].i = 0.;
/* L30: */
		}
/* L40: */
	    }

/*           B(1:N,1:NRHS) := Q(1:N,:)**H * B(1:M,1:NRHS) */

	    i__1 = *lwork - mn;
	    wunmlq_("Left", "Conjugate transpose", n, nrhs, m, &a[a_offset], 
		    lda, &work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, 
		    info);

/*           workspace at least NRHS, optimally NRHS*NB */

	    scllen = *n;

	} else {

/*           overdetermined system f2cmin || A**H * X - B || */

/*           B(1:N,1:NRHS) := Q * B(1:N,1:NRHS) */

	    i__1 = *lwork - mn;
	    wunmlq_("Left", "No transpose", n, nrhs, m, &a[a_offset], lda, &
		    work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);

/*           workspace at least NRHS, optimally NRHS*NB */

/*           B(1:M,1:NRHS) := inv(L**H) * B(1:M,1:NRHS) */

	    wtrtrs_("Lower", "Conjugate transpose", "Non-unit", m, nrhs, &a[
		    a_offset], lda, &b[b_offset], ldb, info);

	    if (*info > 0) {
		return;
	    }

	    scllen = *m;

	}

    }

/*     Undo scaling */

    if (iascl == 1) {
	wlascl_("G", &c__0, &c__0, &anrm, &smlnum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (iascl == 2) {
	wlascl_("G", &c__0, &c__0, &anrm, &bignum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }
    if (ibscl == 1) {
	wlascl_("G", &c__0, &c__0, &smlnum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (ibscl == 2) {
	wlascl_("G", &c__0, &c__0, &bignum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }

L50:
    d__1 = (quadreal) wsize;
    work[1].r = d__1, work[1].i = 0.;

    return;

/*     End of ZGELS */

} /* wgels_ */

