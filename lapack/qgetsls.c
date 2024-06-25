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

static integer c_n1 = -1;
static integer c_n2 = -2;
static quadreal c_b23 = 0.;
static integer c__0 = 0;

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DGETSLS( TRANS, M, N, NRHS, A, LDA, B, LDB, */
/*     $                     WORK, LWORK, INFO ) */

/*       CHARACTER          TRANS */
/*       INTEGER            INFO, LDA, LDB, LWORK, M, N, NRHS */
/*       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGETSLS solves overdetermined or underdetermined doublereal linear systems */
/* > involving an M-by-N matrix A, using a tall skinny QR or short wide LQ */
/* > factorization of A.  It is assumed that A has full rank. */
/* > */
/* > */
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
/* > 3. If TRANS = 'T' and m >= n:  find the minimum norm solution of */
/* >    an undetermined system A**T * X = B. */
/* > */
/* > 4. If TRANS = 'T' and m < n:  find the least squares solution of */
/* >    an overdetermined system, i.e., solve the least squares problem */
/* >                 minimize || B - A**T * X ||. */
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
/* >          = 'T': the linear system involves A**T. */
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
/* >          columns of the matrices B and X. NRHS >=0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA,N) */
/* >          On entry, the M-by-N matrix A. */
/* >          On exit, */
/* >          A is overwritten by details of its QR or LQ */
/* >          factorization as returned by DGEQR or DGELQ. */
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
/* >          B is DOUBLE PRECISION array, dimension (LDB,NRHS) */
/* >          On entry, the matrix B of right hand side vectors, stored */
/* >          columnwise; B is M-by-NRHS if TRANS = 'N', or N-by-NRHS */
/* >          if TRANS = 'T'. */
/* >          On exit, if INFO = 0, B is overwritten by the solution */
/* >          vectors, stored columnwise: */
/* >          if TRANS = 'N' and m >= n, rows 1 to n of B contain the least */
/* >          squares solution vectors. */
/* >          if TRANS = 'N' and m < n, rows 1 to N of B contain the */
/* >          minimum norm solution vectors; */
/* >          if TRANS = 'T' and m >= n, rows 1 to M of B contain the */
/* >          minimum norm solution vectors; */
/* >          if TRANS = 'T' and m < n, rows 1 to M of B contain the */
/* >          least squares solution vectors. */
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
/* >          (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO = 0, WORK(1) contains optimal (or either minimal */
/* >          or optimal, if query was assumed) LWORK. */
/* >          See LWORK for details. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK. */
/* >          If LWORK = -1 or -2, then a workspace query is assumed. */
/* >          If LWORK = -1, the routine calculates optimal size of WORK for the */
/* >          optimal performance and returns this value in WORK(1). */
/* >          If LWORK = -2, the routine calculates minimal size of WORK and */
/* >          returns this value in WORK(1). */
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

/* > \date June 2017 */

/* > \ingroup doubleGEsolve */

/*  ===================================================================== */
void  qgetsls_(char *trans, integer *m, integer *n, integer *
	nrhs, quadreal *a, integer *lda, quadreal *b, integer *ldb, 
	quadreal *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    integer i__, j;
    quadreal tq[5];
    integer lw1, lw2, mnk, lwm, lwo;
    quadreal anrm, bnrm;
    logical tran;
    integer brow, tszm, tszo, info2, iascl, ibscl;
    extern void  qgelq_(integer *, integer *, quadreal *, 
	    integer *, quadreal *, integer *, quadreal *, integer *, 
	    integer *);
    extern logical lsame_(char *, char *);
    extern void  qgeqr_(integer *, integer *, quadreal *, 
	    integer *, quadreal *, integer *, quadreal *, integer *, 
	    integer *);
    integer minmn, maxmn;
    quadreal workq[1];
    extern void  qlabad_(quadreal *, quadreal *);
    extern quadreal qlamch_(char *), qlange_(char *, integer *, 
	    integer *, quadreal *, integer *, quadreal *);
    extern void  qlascl_(char *, integer *, integer *, 
	    quadreal *, quadreal *, integer *, integer *, quadreal *, 
	    integer *, integer *), qgemlq_(char *, char *, integer *, 
	    integer *, integer *, quadreal *, integer *, quadreal *, 
	    integer *, quadreal *, integer *, quadreal *, integer *, 
	    integer *), qlaset_(char *, integer *, integer *, 
	    quadreal *, quadreal *, quadreal *, integer *), 
	    xerbla_(char *, integer *), qgemqr_(char *, char *, 
	    integer *, integer *, integer *, quadreal *, integer *, 
	    quadreal *, integer *, quadreal *, integer *, quadreal *, 
	    integer *, integer *);
    integer scllen;
    quadreal bignum, smlnum;
    integer wsizem, wsizeo;
    logical lquery;
    extern void  qtrtrs_(char *, char *, char *, integer *, 
	    integer *, quadreal *, integer *, quadreal *, integer *, 
	    integer *);


/*  -- LAPACK driver routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */



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
    minmn = f2cmin(*m,*n);
    maxmn = f2cmax(*m,*n);
    mnk = f2cmax(minmn,*nrhs);
    tran = lsame_(trans, "T");

    lquery = *lwork == -1 || *lwork == -2;
    if (! (lsame_(trans, "N") || lsame_(trans, "T"))) {
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
	}
    }

    if (*info == 0) {

/*     Determine the block size and minimum LWORK */

	if (*m >= *n) {
	    qgeqr_(m, n, &a[a_offset], lda, tq, &c_n1, workq, &c_n1, &info2);
	    tszo = (integer) tq[0];
	    lwo = (integer) workq[0];
	    qgemqr_("L", trans, m, nrhs, n, &a[a_offset], lda, tq, &tszo, &b[
		    b_offset], ldb, workq, &c_n1, &info2);
/* Computing MAX */
	    i__1 = lwo, i__2 = (integer) workq[0];
	    lwo = f2cmax(i__1,i__2);
	    qgeqr_(m, n, &a[a_offset], lda, tq, &c_n2, workq, &c_n2, &info2);
	    tszm = (integer) tq[0];
	    lwm = (integer) workq[0];
	    qgemqr_("L", trans, m, nrhs, n, &a[a_offset], lda, tq, &tszm, &b[
		    b_offset], ldb, workq, &c_n1, &info2);
/* Computing MAX */
	    i__1 = lwm, i__2 = (integer) workq[0];
	    lwm = f2cmax(i__1,i__2);
	    wsizeo = tszo + lwo;
	    wsizem = tszm + lwm;
	} else {
	    qgelq_(m, n, &a[a_offset], lda, tq, &c_n1, workq, &c_n1, &info2);
	    tszo = (integer) tq[0];
	    lwo = (integer) workq[0];
	    qgemlq_("L", trans, n, nrhs, m, &a[a_offset], lda, tq, &tszo, &b[
		    b_offset], ldb, workq, &c_n1, &info2);
/* Computing MAX */
	    i__1 = lwo, i__2 = (integer) workq[0];
	    lwo = f2cmax(i__1,i__2);
	    qgelq_(m, n, &a[a_offset], lda, tq, &c_n2, workq, &c_n2, &info2);
	    tszm = (integer) tq[0];
	    lwm = (integer) workq[0];
	    qgemlq_("L", trans, n, nrhs, m, &a[a_offset], lda, tq, &tszo, &b[
		    b_offset], ldb, workq, &c_n1, &info2);
/* Computing MAX */
	    i__1 = lwm, i__2 = (integer) workq[0];
	    lwm = f2cmax(i__1,i__2);
	    wsizeo = tszo + lwo;
	    wsizem = tszm + lwm;
	}

	if (*lwork < wsizem && ! lquery) {
	    *info = -10;
	}

    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGETSLS", &i__1);
	work[1] = (quadreal) wsizeo;
	return;
    }
    if (lquery) {
	if (*lwork == -1) {
	    work[1] = (doublereal) wsizeo;
	}
	if (*lwork == -2) {
	    work[1] = (doublereal) wsizem;
	}
	return;
    }
    if (*lwork < wsizeo) {
	lw1 = tszm;
	lw2 = lwm;
    } else {
	lw1 = tszo;
	lw2 = lwo;
    }

/*     Quick return if possible */

/* Computing MIN */
    i__1 = f2cmin(*m,*n);
    if (f2cmin(i__1,*nrhs) == 0) {
	i__1 = f2cmax(*m,*n);
	qlaset_("FULL", &i__1, nrhs, &c_b23, &c_b23, &b[b_offset], ldb);
	return;
    }

/*     Get machine parameters */

    smlnum = qlamch_("S") / qlamch_("P");
    bignum = 1. / smlnum;
    qlabad_(&smlnum, &bignum);

/*     Scale A, B if f2cmax element outside range [SMLNUM,BIGNUM] */

    anrm = qlange_("M", m, n, &a[a_offset], lda, &work[1]);
    iascl = 0;
    if (anrm > 0. && anrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	qlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, 
		info);
	iascl = 1;
    } else if (anrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	qlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, 
		info);
	iascl = 2;
    } else if (anrm == 0.) {

/*        Matrix all zero. Return zero solution. */

	qlaset_("F", &maxmn, nrhs, &c_b23, &c_b23, &b[b_offset], ldb);
	goto L50;
    }

    brow = *m;
    if (tran) {
	brow = *n;
    }
    bnrm = qlange_("M", &brow, nrhs, &b[b_offset], ldb, &work[1]);
    ibscl = 0;
    if (bnrm > 0. && bnrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	qlascl_("G", &c__0, &c__0, &bnrm, &smlnum, &brow, nrhs, &b[b_offset], 
		ldb, info);
	ibscl = 1;
    } else if (bnrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	qlascl_("G", &c__0, &c__0, &bnrm, &bignum, &brow, nrhs, &b[b_offset], 
		ldb, info);
	ibscl = 2;
    }

    if (*m >= *n) {

/*        compute QR factorization of A */

	qgeqr_(m, n, &a[a_offset], lda, &work[lw2 + 1], &lw1, &work[1], &lw2, 
		info);
	if (! tran) {

/*           Least-Squares Problem f2cmin || A * X - B || */

/*           B(1:M,1:NRHS) := Q**T * B(1:M,1:NRHS) */

	    qgemqr_("L", "T", m, nrhs, n, &a[a_offset], lda, &work[lw2 + 1], &
		    lw1, &b[b_offset], ldb, &work[1], &lw2, info);

/*           B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS) */

	    qtrtrs_("U", "N", "N", n, nrhs, &a[a_offset], lda, &b[b_offset], 
		    ldb, info);
	    if (*info > 0) {
		return;
	    }
	    scllen = *n;
	} else {

/*           Overdetermined system of equations A**T * X = B */

/*           B(1:N,1:NRHS) := inv(R**T) * B(1:N,1:NRHS) */

	    qtrtrs_("U", "T", "N", n, nrhs, &a[a_offset], lda, &b[b_offset], 
		    ldb, info);

	    if (*info > 0) {
		return;
	    }

/*           B(N+1:M,1:NRHS) = ZERO */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = *n + 1; i__ <= i__2; ++i__) {
		    b[i__ + j * b_dim1] = 0.;
/* L10: */
		}
/* L20: */
	    }

/*           B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS) */

	    qgemqr_("L", "N", m, nrhs, n, &a[a_offset], lda, &work[lw2 + 1], &
		    lw1, &b[b_offset], ldb, &work[1], &lw2, info);

	    scllen = *m;

	}

    } else {

/*        Compute LQ factorization of A */

	qgelq_(m, n, &a[a_offset], lda, &work[lw2 + 1], &lw1, &work[1], &lw2, 
		info);

/*        workspace at least M, optimally M*NB. */

	if (! tran) {

/*           underdetermined system of equations A * X = B */

/*           B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS) */

	    qtrtrs_("L", "N", "N", m, nrhs, &a[a_offset], lda, &b[b_offset], 
		    ldb, info);

	    if (*info > 0) {
		return;
	    }

/*           B(M+1:N,1:NRHS) = 0 */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = *m + 1; i__ <= i__2; ++i__) {
		    b[i__ + j * b_dim1] = 0.;
/* L30: */
		}
/* L40: */
	    }

/*           B(1:N,1:NRHS) := Q(1:N,:)**T * B(1:M,1:NRHS) */

	    qgemlq_("L", "T", n, nrhs, m, &a[a_offset], lda, &work[lw2 + 1], &
		    lw1, &b[b_offset], ldb, &work[1], &lw2, info);

/*           workspace at least NRHS, optimally NRHS*NB */

	    scllen = *n;

	} else {

/*           overdetermined system f2cmin || A**T * X - B || */

/*           B(1:N,1:NRHS) := Q * B(1:N,1:NRHS) */

	    qgemlq_("L", "N", n, nrhs, m, &a[a_offset], lda, &work[lw2 + 1], &
		    lw1, &b[b_offset], ldb, &work[1], &lw2, info);

/*           workspace at least NRHS, optimally NRHS*NB */

/*           B(1:M,1:NRHS) := inv(L**T) * B(1:M,1:NRHS) */

	    qtrtrs_("Lower", "Transpose", "Non-unit", m, nrhs, &a[a_offset], 
		    lda, &b[b_offset], ldb, info);

	    if (*info > 0) {
		return;
	    }

	    scllen = *m;

	}

    }

/*     Undo scaling */

    if (iascl == 1) {
	qlascl_("G", &c__0, &c__0, &anrm, &smlnum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (iascl == 2) {
	qlascl_("G", &c__0, &c__0, &anrm, &bignum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }
    if (ibscl == 1) {
	qlascl_("G", &c__0, &c__0, &smlnum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (ibscl == 2) {
	qlascl_("G", &c__0, &c__0, &bignum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }

L50:
    work[1] = (quadreal) (tszo + lwo);
    return;

/*     End of DGETSLS */

} /* qgetsls_ */

