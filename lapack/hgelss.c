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

static integer c__6 = 6;
static integer c_n1 = -1;
static integer c__0 = 0;
static halfreal c_b46 = 0.;
static integer c__1 = 1;
static halfreal c_b79 = 1.;

/* > \brief <b> DGELSS solves overdetermined or underdetermined systems for GE matrices</b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DGELSS + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgelss.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgelss.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgelss.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, */
/*                          WORK, LWORK, INFO ) */

/*       INTEGER            INFO, LDA, LDB, LWORK, M, N, NRHS, RANK */
/*       DOUBLE PRECISION   RCOND */
/*       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), S( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGELSS computes the minimum norm solution to a doublereal linear least */
/* > squares problem: */
/* > */
/* > Minimize 2-norm(| b - A*x |). */
/* > */
/* > using the singular value decomposition (SVD) of A. A is an M-by-N */
/* > matrix which may be rank-deficient. */
/* > */
/* > Several right hand side vectors b and solution vectors x can be */
/* > handled in a single call; they are stored as the columns of the */
/* > M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix */
/* > X. */
/* > */
/* > The effective rank of A is determined by treating as zero those */
/* > singular values which are less than RCOND times the largest singular */
/* > value. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A. M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right hand sides, i.e., the number of columns */
/* >          of the matrices B and X. NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA,N) */
/* >          On entry, the M-by-N matrix A. */
/* >          On exit, the first f2cmin(m,n) rows of A are overwritten with */
/* >          its right singular vectors, stored rowwise. */
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
/* >          On entry, the M-by-NRHS right hand side matrix B. */
/* >          On exit, B is overwritten by the N-by-NRHS solution */
/* >          matrix X.  If m >= n and RANK = n, the residual */
/* >          sum-of-squares for the solution in the i-th column is given */
/* >          by the sum of squares of elements n+1:m in that column. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. LDB >= f2cmax(1,f2cmax(M,N)). */
/* > \endverbatim */
/* > */
/* > \param[out] S */
/* > \verbatim */
/* >          S is DOUBLE PRECISION array, dimension (f2cmin(M,N)) */
/* >          The singular values of A in decreasing order. */
/* >          The condition number of A in the 2-norm = S(1)/S(f2cmin(m,n)). */
/* > \endverbatim */
/* > */
/* > \param[in] RCOND */
/* > \verbatim */
/* >          RCOND is DOUBLE PRECISION */
/* >          RCOND is used to determine the effective rank of A. */
/* >          Singular values S(i) <= RCOND*S(1) are treated as zero. */
/* >          If RCOND < 0, machine precision is used instead. */
/* > \endverbatim */
/* > */
/* > \param[out] RANK */
/* > \verbatim */
/* >          RANK is INTEGER */
/* >          The effective rank of A, i.e., the number of singular values */
/* >          which are greater than RCOND*S(1). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK. LWORK >= 1, and also: */
/* >          LWORK >= 3*f2cmin(M,N) + f2cmax( 2*f2cmin(M,N), f2cmax(M,N), NRHS ) */
/* >          For good performance, LWORK should generally be larger. */
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
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* >          > 0:  the algorithm for computing the SVD failed to converge; */
/* >                if INFO = i, i off-diagonal elements of an intermediate */
/* >                bidiagonal form did not converge to zero. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleGEsolve */

/*  ===================================================================== */
void  hgelss_(integer *m, integer *n, integer *nrhs, 
	halfreal *a, integer *lda, halfreal *b, integer *ldb, halfreal *
	s, halfreal *rcond, integer *rank, halfreal *work, integer *lwork,
	 integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4;
    halfreal d__1;

    /* Local variables */
    integer i__, bl, ie, il, mm;
    halfreal dum[1], eps, thr, anrm, bnrm;
    integer itau, lwork_dgebrd__, lwork_dgelqf__, lwork_dgeqrf__, 
	    lwork_dorgbr__, lwork_dormbr__, lwork_dormlq__, lwork_dormqr__;
    extern void  hgemm_(char *, char *, integer *, integer *, 
	    integer *, halfreal *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, integer *);
    integer iascl, ibscl;
    extern void  hgemv_(char *, integer *, integer *, 
	    halfreal *, halfreal *, integer *, halfreal *, integer *, 
	    halfreal *, halfreal *, integer *), hrscl_(integer *, 
	    halfreal *, halfreal *, integer *);
    integer chunk;
    halfreal sfmin;
    integer minmn;
    extern void  hcopy_(integer *, halfreal *, integer *, 
	    halfreal *, integer *);
    integer maxmn, itaup, itauq, mnthr, iwork;
    extern void  hlabad_(halfreal *, halfreal *), hgebrd_(
	    integer *, integer *, halfreal *, integer *, halfreal *, 
	    halfreal *, halfreal *, halfreal *, halfreal *, integer *,
	     integer *);
    extern halfreal hlamch_(char *), hlange_(char *, integer *, 
	    integer *, halfreal *, integer *, halfreal *);
    integer bdspac;
    extern void  hgelqf_(integer *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, integer *, integer *), 
	    hlascl_(char *, integer *, integer *, halfreal *, halfreal *, 
	    integer *, integer *, halfreal *, integer *, integer *),
	     hgeqrf_(integer *, integer *, halfreal *, integer *, 
	    halfreal *, halfreal *, integer *, integer *), hlacpy_(char *,
	     integer *, integer *, halfreal *, integer *, halfreal *, 
	    integer *), hlaset_(char *, integer *, integer *, 
	    halfreal *, halfreal *, halfreal *, integer *), 
	    xerbla_(char *, integer *), hbdsqr_(char *, integer *, 
	    integer *, integer *, integer *, halfreal *, halfreal *, 
	    halfreal *, integer *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, integer *), horgbr_(char *, 
	    integer *, integer *, integer *, halfreal *, integer *, 
	    halfreal *, halfreal *, integer *, integer *);
    halfreal bignum;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern void  hormbr_(char *, char *, char *, integer *, 
	    integer *, integer *, halfreal *, integer *, halfreal *, 
	    halfreal *, integer *, halfreal *, integer *, integer *), hormlq_(char *, char *, integer *, 
	    integer *, integer *, halfreal *, integer *, halfreal *, 
	    halfreal *, integer *, halfreal *, integer *, integer *);
    integer ldwork;
    extern void  hormqr_(char *, char *, integer *, integer *, 
	    integer *, halfreal *, integer *, halfreal *, halfreal *, 
	    integer *, halfreal *, integer *, integer *);
    integer minwrk, maxwrk;
    halfreal smlnum;
    logical lquery;


/*  -- LAPACK driver routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --s;
    --work;

    /* Function Body */
    *info = 0;
    minmn = f2cmin(*m,*n);
    maxmn = f2cmax(*m,*n);
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -5;
    } else if (*ldb < f2cmax(1,maxmn)) {
	*info = -7;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       NB refers to the optimal block size for the immediately */
/*       following subroutine, as returned by ILAENV.) */

    if (*info == 0) {
	minwrk = 1;
	maxwrk = 1;
	if (minmn > 0) {
	    mm = *m;
	    mnthr = ilaenv_(&c__6, "DGELSS", " ", m, n, nrhs, &c_n1, (ftnlen)
		    6, (ftnlen)1);
	    if (*m >= *n && *m >= mnthr) {

/*              Path 1a - overdetermined, with many more rows than */
/*                        columns */

/*              Compute space needed for DGEQRF */
		hgeqrf_(m, n, &a[a_offset], lda, dum, dum, &c_n1, info);
		lwork_dgeqrf__ = (integer) dum[0];
/*              Compute space needed for DORMQR */
		hormqr_("L", "T", m, nrhs, n, &a[a_offset], lda, dum, &b[
			b_offset], ldb, dum, &c_n1, info);
		lwork_dormqr__ = (integer) dum[0];
		mm = *n;
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n + lwork_dgeqrf__;
		maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n + lwork_dormqr__;
		maxwrk = f2cmax(i__1,i__2);
	    }
	    if (*m >= *n) {

/*              Path 1 - overdetermined or exactly determined */

/*              Compute workspace needed for DBDSQR */

/* Computing MAX */
		i__1 = 1, i__2 = *n * 5;
		bdspac = f2cmax(i__1,i__2);
/*              Compute space needed for DGEBRD */
		hgebrd_(&mm, n, &a[a_offset], lda, &s[1], dum, dum, dum, dum, 
			&c_n1, info);
		lwork_dgebrd__ = (integer) dum[0];
/*              Compute space needed for DORMBR */
		hormbr_("Q", "L", "T", &mm, nrhs, n, &a[a_offset], lda, dum, &
			b[b_offset], ldb, dum, &c_n1, info);
		lwork_dormbr__ = (integer) dum[0];
/*              Compute space needed for DORGBR */
		horgbr_("P", n, n, n, &a[a_offset], lda, dum, dum, &c_n1, 
			info);
		lwork_dorgbr__ = (integer) dum[0];
/*              Compute total workspace needed */
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n * 3 + lwork_dgebrd__;
		maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n * 3 + lwork_dormbr__;
		maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n * 3 + lwork_dorgbr__;
		maxwrk = f2cmax(i__1,i__2);
		maxwrk = f2cmax(maxwrk,bdspac);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n * *nrhs;
		maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		i__1 = *n * 3 + mm, i__2 = *n * 3 + *nrhs, i__1 = f2cmax(i__1,
			i__2);
		minwrk = f2cmax(i__1,bdspac);
		maxwrk = f2cmax(minwrk,maxwrk);
	    }
	    if (*n > *m) {

/*              Compute workspace needed for DBDSQR */

/* Computing MAX */
		i__1 = 1, i__2 = *m * 5;
		bdspac = f2cmax(i__1,i__2);
/* Computing MAX */
		i__1 = *m * 3 + *nrhs, i__2 = *m * 3 + *n, i__1 = f2cmax(i__1,
			i__2);
		minwrk = f2cmax(i__1,bdspac);
		if (*n >= mnthr) {

/*                 Path 2a - underdetermined, with many more columns */
/*                 than rows */

/*                 Compute space needed for DGELQF */
		    hgelqf_(m, n, &a[a_offset], lda, dum, dum, &c_n1, info);
		    lwork_dgelqf__ = (integer) dum[0];
/*                 Compute space needed for DGEBRD */
		    hgebrd_(m, m, &a[a_offset], lda, &s[1], dum, dum, dum, 
			    dum, &c_n1, info);
		    lwork_dgebrd__ = (integer) dum[0];
/*                 Compute space needed for DORMBR */
		    hormbr_("Q", "L", "T", m, nrhs, n, &a[a_offset], lda, dum,
			     &b[b_offset], ldb, dum, &c_n1, info);
		    lwork_dormbr__ = (integer) dum[0];
/*                 Compute space needed for DORGBR */
		    horgbr_("P", m, m, m, &a[a_offset], lda, dum, dum, &c_n1, 
			    info);
		    lwork_dorgbr__ = (integer) dum[0];
/*                 Compute space needed for DORMLQ */
		    hormlq_("L", "T", n, nrhs, m, &a[a_offset], lda, dum, &b[
			    b_offset], ldb, dum, &c_n1, info);
		    lwork_dormlq__ = (integer) dum[0];
/*                 Compute total workspace needed */
		    maxwrk = *m + lwork_dgelqf__;
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + (*m << 2) + 
			    lwork_dgebrd__;
		    maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + (*m << 2) + 
			    lwork_dormbr__;
		    maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + (*m << 2) + 
			    lwork_dorgbr__;
		    maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + *m + bdspac;
		    maxwrk = f2cmax(i__1,i__2);
		    if (*nrhs > 1) {
/* Computing MAX */
			i__1 = maxwrk, i__2 = *m * *m + *m + *m * *nrhs;
			maxwrk = f2cmax(i__1,i__2);
		    } else {
/* Computing MAX */
			i__1 = maxwrk, i__2 = *m * *m + (*m << 1);
			maxwrk = f2cmax(i__1,i__2);
		    }
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m + lwork_dormlq__;
		    maxwrk = f2cmax(i__1,i__2);
		} else {

/*                 Path 2 - underdetermined */

/*                 Compute space needed for DGEBRD */
		    hgebrd_(m, n, &a[a_offset], lda, &s[1], dum, dum, dum, 
			    dum, &c_n1, info);
		    lwork_dgebrd__ = (integer) dum[0];
/*                 Compute space needed for DORMBR */
		    hormbr_("Q", "L", "T", m, nrhs, m, &a[a_offset], lda, dum,
			     &b[b_offset], ldb, dum, &c_n1, info);
		    lwork_dormbr__ = (integer) dum[0];
/*                 Compute space needed for DORGBR */
		    horgbr_("P", m, n, m, &a[a_offset], lda, dum, dum, &c_n1, 
			    info);
		    lwork_dorgbr__ = (integer) dum[0];
		    maxwrk = *m * 3 + lwork_dgebrd__;
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * 3 + lwork_dormbr__;
		    maxwrk = f2cmax(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * 3 + lwork_dorgbr__;
		    maxwrk = f2cmax(i__1,i__2);
		    maxwrk = f2cmax(maxwrk,bdspac);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *n * *nrhs;
		    maxwrk = f2cmax(i__1,i__2);
		}
	    }
	    maxwrk = f2cmax(minwrk,maxwrk);
	}
	work[1] = (halfreal) maxwrk;

	if (*lwork < minwrk && ! lquery) {
	    *info = -12;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGELSS", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	*rank = 0;
	return;
    }

/*     Get machine parameters */

    eps = hlamch_("P");
    sfmin = hlamch_("S");
    smlnum = sfmin / eps;
    bignum = 1. / smlnum;
    hlabad_(&smlnum, &bignum);

/*     Scale A if f2cmax element outside range [SMLNUM,BIGNUM] */

    anrm = hlange_("M", m, n, &a[a_offset], lda, &work[1]);
    iascl = 0;
    if (anrm > 0. && anrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	hlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, 
		info);
	iascl = 1;
    } else if (anrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	hlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, 
		info);
	iascl = 2;
    } else if (anrm == 0.) {

/*        Matrix all zero. Return zero solution. */

	i__1 = f2cmax(*m,*n);
	hlaset_("F", &i__1, nrhs, &c_b46, &c_b46, &b[b_offset], ldb);
	hlaset_("F", &minmn, &c__1, &c_b46, &c_b46, &s[1], &minmn);
	*rank = 0;
	goto L70;
    }

/*     Scale B if f2cmax element outside range [SMLNUM,BIGNUM] */

    bnrm = hlange_("M", m, nrhs, &b[b_offset], ldb, &work[1]);
    ibscl = 0;
    if (bnrm > 0. && bnrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	hlascl_("G", &c__0, &c__0, &bnrm, &smlnum, m, nrhs, &b[b_offset], ldb,
		 info);
	ibscl = 1;
    } else if (bnrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	hlascl_("G", &c__0, &c__0, &bnrm, &bignum, m, nrhs, &b[b_offset], ldb,
		 info);
	ibscl = 2;
    }

/*     Overdetermined case */

    if (*m >= *n) {

/*        Path 1 - overdetermined or exactly determined */

	mm = *m;
	if (*m >= mnthr) {

/*           Path 1a - overdetermined, with many more rows than columns */

	    mm = *n;
	    itau = 1;
	    iwork = itau + *n;

/*           Compute A=Q*R */
/*           (Workspace: need 2*N, prefer N+N*NB) */

	    i__1 = *lwork - iwork + 1;
	    hgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[iwork], &i__1,
		     info);

/*           Multiply B by transpose(Q) */
/*           (Workspace: need N+NRHS, prefer N+NRHS*NB) */

	    i__1 = *lwork - iwork + 1;
	    hormqr_("L", "T", m, nrhs, n, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[iwork], &i__1, info);

/*           Zero out below R */

	    if (*n > 1) {
		i__1 = *n - 1;
		i__2 = *n - 1;
		hlaset_("L", &i__1, &i__2, &c_b46, &c_b46, &a[a_dim1 + 2], 
			lda);
	    }
	}

	ie = 1;
	itauq = ie + *n;
	itaup = itauq + *n;
	iwork = itaup + *n;

/*        Bidiagonalize R in A */
/*        (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB) */

	i__1 = *lwork - iwork + 1;
	hgebrd_(&mm, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		work[itaup], &work[iwork], &i__1, info);

/*        Multiply B by transpose of left bidiagonalizing vectors of R */
/*        (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB) */

	i__1 = *lwork - iwork + 1;
	hormbr_("Q", "L", "T", &mm, nrhs, n, &a[a_offset], lda, &work[itauq], 
		&b[b_offset], ldb, &work[iwork], &i__1, info);

/*        Generate right bidiagonalizing vectors of R in A */
/*        (Workspace: need 4*N-1, prefer 3*N+(N-1)*NB) */

	i__1 = *lwork - iwork + 1;
	horgbr_("P", n, n, n, &a[a_offset], lda, &work[itaup], &work[iwork], &
		i__1, info);
	iwork = ie + *n;

/*        Perform bidiagonal QR iteration */
/*          multiply B by transpose of left singular vectors */
/*          compute right singular vectors in A */
/*        (Workspace: need BDSPAC) */

	hbdsqr_("U", n, n, &c__0, nrhs, &s[1], &work[ie], &a[a_offset], lda, 
		dum, &c__1, &b[b_offset], ldb, &work[iwork], info);
	if (*info != 0) {
	    goto L70;
	}

/*        Multiply B by reciprocals of singular values */

/* Computing MAX */
	d__1 = *rcond * s[1];
	thr = f2cmax(d__1,sfmin);
	if (*rcond < 0.) {
/* Computing MAX */
	    d__1 = eps * s[1];
	    thr = f2cmax(d__1,sfmin);
	}
	*rank = 0;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (s[i__] > thr) {
		hrscl_(nrhs, &s[i__], &b[i__ + b_dim1], ldb);
		++(*rank);
	    } else {
		hlaset_("F", &c__1, nrhs, &c_b46, &c_b46, &b[i__ + b_dim1], 
			ldb);
	    }
/* L10: */
	}

/*        Multiply B by right singular vectors */
/*        (Workspace: need N, prefer N*NRHS) */

	if (*lwork >= *ldb * *nrhs && *nrhs > 1) {
	    hgemm_("T", "N", n, nrhs, n, &c_b79, &a[a_offset], lda, &b[
		    b_offset], ldb, &c_b46, &work[1], ldb);
	    hlacpy_("G", n, nrhs, &work[1], ldb, &b[b_offset], ldb)
		    ;
	} else if (*nrhs > 1) {
	    chunk = *lwork / *n;
	    i__1 = *nrhs;
	    i__2 = chunk;
	    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
		i__3 = *nrhs - i__ + 1;
		bl = f2cmin(i__3,chunk);
		hgemm_("T", "N", n, &bl, n, &c_b79, &a[a_offset], lda, &b[i__ 
			* b_dim1 + 1], ldb, &c_b46, &work[1], n);
		hlacpy_("G", n, &bl, &work[1], n, &b[i__ * b_dim1 + 1], ldb);
/* L20: */
	    }
	} else {
	    hgemv_("T", n, n, &c_b79, &a[a_offset], lda, &b[b_offset], &c__1, 
		    &c_b46, &work[1], &c__1);
	    hcopy_(n, &work[1], &c__1, &b[b_offset], &c__1);
	}

    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__2 = *m, i__1 = (*m << 1) - 4, i__2 = f2cmax(i__2,i__1), i__2 = f2cmax(
		i__2,*nrhs), i__1 = *n - *m * 3;
	if (*n >= mnthr && *lwork >= (*m << 2) + *m * *m + f2cmax(i__2,i__1)) {

/*        Path 2a - underdetermined, with many more columns than rows */
/*        and sufficient workspace for an efficient algorithm */

	    ldwork = *m;
/* Computing MAX */
/* Computing MAX */
	    i__3 = *m, i__4 = (*m << 1) - 4, i__3 = f2cmax(i__3,i__4), i__3 = 
		    f2cmax(i__3,*nrhs), i__4 = *n - *m * 3;
	    i__2 = (*m << 2) + *m * *lda + f2cmax(i__3,i__4), i__1 = *m * *lda + 
		    *m + *m * *nrhs;
	    if (*lwork >= f2cmax(i__2,i__1)) {
		ldwork = *lda;
	    }
	    itau = 1;
	    iwork = *m + 1;

/*        Compute A=L*Q */
/*        (Workspace: need 2*M, prefer M+M*NB) */

	    i__2 = *lwork - iwork + 1;
	    hgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[iwork], &i__2,
		     info);
	    il = iwork;

/*        Copy L to WORK(IL), zeroing out above it */

	    hlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwork);
	    i__2 = *m - 1;
	    i__1 = *m - 1;
	    hlaset_("U", &i__2, &i__1, &c_b46, &c_b46, &work[il + ldwork], &
		    ldwork);
	    ie = il + ldwork * *m;
	    itauq = ie + *m;
	    itaup = itauq + *m;
	    iwork = itaup + *m;

/*        Bidiagonalize L in WORK(IL) */
/*        (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB) */

	    i__2 = *lwork - iwork + 1;
	    hgebrd_(m, m, &work[il], &ldwork, &s[1], &work[ie], &work[itauq], 
		    &work[itaup], &work[iwork], &i__2, info);

/*        Multiply B by transpose of left bidiagonalizing vectors of L */
/*        (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB) */

	    i__2 = *lwork - iwork + 1;
	    hormbr_("Q", "L", "T", m, nrhs, m, &work[il], &ldwork, &work[
		    itauq], &b[b_offset], ldb, &work[iwork], &i__2, info);

/*        Generate right bidiagonalizing vectors of R in WORK(IL) */
/*        (Workspace: need M*M+5*M-1, prefer M*M+4*M+(M-1)*NB) */

	    i__2 = *lwork - iwork + 1;
	    horgbr_("P", m, m, m, &work[il], &ldwork, &work[itaup], &work[
		    iwork], &i__2, info);
	    iwork = ie + *m;

/*        Perform bidiagonal QR iteration, */
/*           computing right singular vectors of L in WORK(IL) and */
/*           multiplying B by transpose of left singular vectors */
/*        (Workspace: need M*M+M+BDSPAC) */

	    hbdsqr_("U", m, m, &c__0, nrhs, &s[1], &work[ie], &work[il], &
		    ldwork, &a[a_offset], lda, &b[b_offset], ldb, &work[iwork]
		    , info);
	    if (*info != 0) {
		goto L70;
	    }

/*        Multiply B by reciprocals of singular values */

/* Computing MAX */
	    d__1 = *rcond * s[1];
	    thr = f2cmax(d__1,sfmin);
	    if (*rcond < 0.) {
/* Computing MAX */
		d__1 = eps * s[1];
		thr = f2cmax(d__1,sfmin);
	    }
	    *rank = 0;
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		if (s[i__] > thr) {
		    hrscl_(nrhs, &s[i__], &b[i__ + b_dim1], ldb);
		    ++(*rank);
		} else {
		    hlaset_("F", &c__1, nrhs, &c_b46, &c_b46, &b[i__ + b_dim1]
			    , ldb);
		}
/* L30: */
	    }
	    iwork = ie;

/*        Multiply B by right singular vectors of L in WORK(IL) */
/*        (Workspace: need M*M+2*M, prefer M*M+M+M*NRHS) */

	    if (*lwork >= *ldb * *nrhs + iwork - 1 && *nrhs > 1) {
		hgemm_("T", "N", m, nrhs, m, &c_b79, &work[il], &ldwork, &b[
			b_offset], ldb, &c_b46, &work[iwork], ldb);
		hlacpy_("G", m, nrhs, &work[iwork], ldb, &b[b_offset], ldb);
	    } else if (*nrhs > 1) {
		chunk = (*lwork - iwork + 1) / *m;
		i__2 = *nrhs;
		i__1 = chunk;
		for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += 
			i__1) {
/* Computing MIN */
		    i__3 = *nrhs - i__ + 1;
		    bl = f2cmin(i__3,chunk);
		    hgemm_("T", "N", m, &bl, m, &c_b79, &work[il], &ldwork, &
			    b[i__ * b_dim1 + 1], ldb, &c_b46, &work[iwork], m);
		    hlacpy_("G", m, &bl, &work[iwork], m, &b[i__ * b_dim1 + 1]
			    , ldb);
/* L40: */
		}
	    } else {
		hgemv_("T", m, m, &c_b79, &work[il], &ldwork, &b[b_dim1 + 1], 
			&c__1, &c_b46, &work[iwork], &c__1);
		hcopy_(m, &work[iwork], &c__1, &b[b_dim1 + 1], &c__1);
	    }

/*        Zero out below first M rows of B */

	    i__1 = *n - *m;
	    hlaset_("F", &i__1, nrhs, &c_b46, &c_b46, &b[*m + 1 + b_dim1], 
		    ldb);
	    iwork = itau + *m;

/*        Multiply transpose(Q) by B */
/*        (Workspace: need M+NRHS, prefer M+NRHS*NB) */

	    i__1 = *lwork - iwork + 1;
	    hormlq_("L", "T", n, nrhs, m, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[iwork], &i__1, info);

	} else {

/*        Path 2 - remaining underdetermined cases */

	    ie = 1;
	    itauq = ie + *m;
	    itaup = itauq + *m;
	    iwork = itaup + *m;

/*        Bidiagonalize A */
/*        (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB) */

	    i__1 = *lwork - iwork + 1;
	    hgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[iwork], &i__1, info);

/*        Multiply B by transpose of left bidiagonalizing vectors */
/*        (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB) */

	    i__1 = *lwork - iwork + 1;
	    hormbr_("Q", "L", "T", m, nrhs, n, &a[a_offset], lda, &work[itauq]
		    , &b[b_offset], ldb, &work[iwork], &i__1, info);

/*        Generate right bidiagonalizing vectors in A */
/*        (Workspace: need 4*M, prefer 3*M+M*NB) */

	    i__1 = *lwork - iwork + 1;
	    horgbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &work[
		    iwork], &i__1, info);
	    iwork = ie + *m;

/*        Perform bidiagonal QR iteration, */
/*           computing right singular vectors of A in A and */
/*           multiplying B by transpose of left singular vectors */
/*        (Workspace: need BDSPAC) */

	    hbdsqr_("L", m, n, &c__0, nrhs, &s[1], &work[ie], &a[a_offset], 
		    lda, dum, &c__1, &b[b_offset], ldb, &work[iwork], info);
	    if (*info != 0) {
		goto L70;
	    }

/*        Multiply B by reciprocals of singular values */

/* Computing MAX */
	    d__1 = *rcond * s[1];
	    thr = f2cmax(d__1,sfmin);
	    if (*rcond < 0.) {
/* Computing MAX */
		d__1 = eps * s[1];
		thr = f2cmax(d__1,sfmin);
	    }
	    *rank = 0;
	    i__1 = *m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		if (s[i__] > thr) {
		    hrscl_(nrhs, &s[i__], &b[i__ + b_dim1], ldb);
		    ++(*rank);
		} else {
		    hlaset_("F", &c__1, nrhs, &c_b46, &c_b46, &b[i__ + b_dim1]
			    , ldb);
		}
/* L50: */
	    }

/*        Multiply B by right singular vectors of A */
/*        (Workspace: need N, prefer N*NRHS) */

	    if (*lwork >= *ldb * *nrhs && *nrhs > 1) {
		hgemm_("T", "N", n, nrhs, m, &c_b79, &a[a_offset], lda, &b[
			b_offset], ldb, &c_b46, &work[1], ldb);
		hlacpy_("F", n, nrhs, &work[1], ldb, &b[b_offset], ldb);
	    } else if (*nrhs > 1) {
		chunk = *lwork / *n;
		i__1 = *nrhs;
		i__2 = chunk;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += 
			i__2) {
/* Computing MIN */
		    i__3 = *nrhs - i__ + 1;
		    bl = f2cmin(i__3,chunk);
		    hgemm_("T", "N", n, &bl, m, &c_b79, &a[a_offset], lda, &b[
			    i__ * b_dim1 + 1], ldb, &c_b46, &work[1], n);
		    hlacpy_("F", n, &bl, &work[1], n, &b[i__ * b_dim1 + 1], 
			    ldb);
/* L60: */
		}
	    } else {
		hgemv_("T", m, n, &c_b79, &a[a_offset], lda, &b[b_offset], &
			c__1, &c_b46, &work[1], &c__1);
		hcopy_(n, &work[1], &c__1, &b[b_offset], &c__1);
	    }
	}
    }

/*     Undo scaling */

    if (iascl == 1) {
	hlascl_("G", &c__0, &c__0, &anrm, &smlnum, n, nrhs, &b[b_offset], ldb,
		 info);
	hlascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    } else if (iascl == 2) {
	hlascl_("G", &c__0, &c__0, &anrm, &bignum, n, nrhs, &b[b_offset], ldb,
		 info);
	hlascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    }
    if (ibscl == 1) {
	hlascl_("G", &c__0, &c__0, &smlnum, &bnrm, n, nrhs, &b[b_offset], ldb,
		 info);
    } else if (ibscl == 2) {
	hlascl_("G", &c__0, &c__0, &bignum, &bnrm, n, nrhs, &b[b_offset], ldb,
		 info);
    }

L70:
    work[1] = (halfreal) maxwrk;
    return;

/*     End of DGELSS */

} /* hgelss_ */

