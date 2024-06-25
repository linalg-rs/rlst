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

static halfcomplex c_b1 = {0.,0.};
static halfcomplex c_b2 = {1.,0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static integer c__2 = 2;

/* > \brief \b ZTREVC3 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZTREVC3 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ztrevc3
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ztrevc3
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ztrevc3
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZTREVC3( SIDE, HOWMNY, SELECT, N, T, LDT, VL, LDVL, VR, */
/*      $                    LDVR, MM, M, WORK, LWORK, RWORK, LRWORK, INFO) */

/*       CHARACTER          HOWMNY, SIDE */
/*       INTEGER            INFO, LDT, LDVL, LDVR, LWORK, M, MM, N */
/*       LOGICAL            SELECT( * ) */
/*       DOUBLE PRECISION   RWORK( * ) */
/*       COMPLEX*16         T( LDT, * ), VL( LDVL, * ), VR( LDVR, * ), */
/*      $                   WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZTREVC3 computes some or all of the right and/or left eigenvectors of */
/* > a complex upper triangular matrix T. */
/* > Matrices of this type are produced by the Schur factorization of */
/* > a complex general matrix:  A = Q*T*Q**H, as computed by ZHSEQR. */
/* > */
/* > The right eigenvector x and the left eigenvector y of T corresponding */
/* > to an eigenvalue w are defined by: */
/* > */
/* >              T*x = w*x,     (y**H)*T = w*(y**H) */
/* > */
/* > where y**H denotes the conjugate transpose of the vector y. */
/* > The eigenvalues are not input to this routine, but are read directly */
/* > from the diagonal of T. */
/* > */
/* > This routine returns the matrices X and/or Y of right and left */
/* > eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an */
/* > input matrix. If Q is the unitary factor that reduces a matrix A to */
/* > Schur form T, then Q*X and Q*Y are the matrices of right and left */
/* > eigenvectors of A. */
/* > */
/* > This uses a Level 3 BLAS version of the back transformation. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] SIDE */
/* > \verbatim */
/* >          SIDE is CHARACTER*1 */
/* >          = 'R':  compute right eigenvectors only; */
/* >          = 'L':  compute left eigenvectors only; */
/* >          = 'B':  compute both right and left eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] HOWMNY */
/* > \verbatim */
/* >          HOWMNY is CHARACTER*1 */
/* >          = 'A':  compute all right and/or left eigenvectors; */
/* >          = 'B':  compute all right and/or left eigenvectors, */
/* >                  backtransformed using the matrices supplied in */
/* >                  VR and/or VL; */
/* >          = 'S':  compute selected right and/or left eigenvectors, */
/* >                  as indicated by the logical array SELECT. */
/* > \endverbatim */
/* > */
/* > \param[in] SELECT */
/* > \verbatim */
/* >          SELECT is LOGICAL array, dimension (N) */
/* >          If HOWMNY = 'S', SELECT specifies the eigenvectors to be */
/* >          computed. */
/* >          The eigenvector corresponding to the j-th eigenvalue is */
/* >          computed if SELECT(j) = .TRUE.. */
/* >          Not referenced if HOWMNY = 'A' or 'B'. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix T. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] T */
/* > \verbatim */
/* >          T is COMPLEX*16 array, dimension (LDT,N) */
/* >          The upper triangular matrix T.  T is modified, but restored */
/* >          on exit. */
/* > \endverbatim */
/* > */
/* > \param[in] LDT */
/* > \verbatim */
/* >          LDT is INTEGER */
/* >          The leading dimension of the array T. LDT >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] VL */
/* > \verbatim */
/* >          VL is COMPLEX*16 array, dimension (LDVL,MM) */
/* >          On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must */
/* >          contain an N-by-N matrix Q (usually the unitary matrix Q of */
/* >          Schur vectors returned by ZHSEQR). */
/* >          On exit, if SIDE = 'L' or 'B', VL contains: */
/* >          if HOWMNY = 'A', the matrix Y of left eigenvectors of T; */
/* >          if HOWMNY = 'B', the matrix Q*Y; */
/* >          if HOWMNY = 'S', the left eigenvectors of T specified by */
/* >                           SELECT, stored consecutively in the columns */
/* >                           of VL, in the same order as their */
/* >                           eigenvalues. */
/* >          Not referenced if SIDE = 'R'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVL */
/* > \verbatim */
/* >          LDVL is INTEGER */
/* >          The leading dimension of the array VL. */
/* >          LDVL >= 1, and if SIDE = 'L' or 'B', LDVL >= N. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VR */
/* > \verbatim */
/* >          VR is COMPLEX*16 array, dimension (LDVR,MM) */
/* >          On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must */
/* >          contain an N-by-N matrix Q (usually the unitary matrix Q of */
/* >          Schur vectors returned by ZHSEQR). */
/* >          On exit, if SIDE = 'R' or 'B', VR contains: */
/* >          if HOWMNY = 'A', the matrix X of right eigenvectors of T; */
/* >          if HOWMNY = 'B', the matrix Q*X; */
/* >          if HOWMNY = 'S', the right eigenvectors of T specified by */
/* >                           SELECT, stored consecutively in the columns */
/* >                           of VR, in the same order as their */
/* >                           eigenvalues. */
/* >          Not referenced if SIDE = 'L'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVR */
/* > \verbatim */
/* >          LDVR is INTEGER */
/* >          The leading dimension of the array VR. */
/* >          LDVR >= 1, and if SIDE = 'R' or 'B', LDVR >= N. */
/* > \endverbatim */
/* > */
/* > \param[in] MM */
/* > \verbatim */
/* >          MM is INTEGER */
/* >          The number of columns in the arrays VL and/or VR. MM >= M. */
/* > \endverbatim */
/* > */
/* > \param[out] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of columns in the arrays VL and/or VR actually */
/* >          used to store the eigenvectors. */
/* >          If HOWMNY = 'A' or 'B', M is set to N. */
/* >          Each selected eigenvector occupies one column. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (MAX(1,LWORK)) */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of array WORK. LWORK >= f2cmax(1,2*N). */
/* >          For optimum performance, LWORK >= N + 2*N*NB, where NB is */
/* >          the optimal blocksize. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the WORK array, returns */
/* >          this value as the first entry of the WORK array, and no error */
/* >          message related to LWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (LRWORK) */
/* > \endverbatim */
/* > */
/* > \param[in] LRWORK */
/* > \verbatim */
/* >          LRWORK is INTEGER */
/* >          The dimension of array RWORK. LRWORK >= f2cmax(1,N). */
/* > */
/* >          If LRWORK = -1, then a workspace query is assumed; the routine */
/* >          only calculates the optimal size of the RWORK array, returns */
/* >          this value as the first entry of the RWORK array, and no error */
/* >          message related to LRWORK is issued by XERBLA. */
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

/*  @precisions fortran z -> c */

/* > \ingroup complex16OTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The algorithm used in this program is basically backward (forward) */
/* >  substitution, with scaling to make the the code robust against */
/* >  possible overflow. */
/* > */
/* >  Each eigenvector is normalized so that the element of largest */
/* >  magnitude has magnitude 1; here the magnitude of a complex number */
/* >  (x,y) is taken to be |x| + |y|. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  ktrevc3_(char *side, char *howmny, logical *select, 
	integer *n, halfcomplex *t, integer *ldt, halfcomplex *vl, 
	integer *ldvl, halfcomplex *vr, integer *ldvr, integer *mm, integer 
	*m, halfcomplex *work, integer *lwork, halfreal *rwork, integer *
	lrwork, integer *info)
{
    /* System generated locals */
    address a__1[2];
    integer t_dim1, t_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1, 
	    i__2[2], i__3, i__4, i__5, i__6;
    halfreal d__1, d__2, d__3;
    halfcomplex z__1, z__2;
    char ch__1[2];

    /* Local variables */
    integer i__, j, k, nb, ii, ki, is, iv;
    halfreal ulp;
    logical allv;
    halfreal unfl, ovfl, smin;
    logical over;
    halfreal scale;
    extern logical lsame_(char *, char *);
    halfreal remax;
    extern void  kgemm_(char *, char *, integer *, integer *, 
	    integer *, halfcomplex *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, halfcomplex *, 
	    integer *);
    logical leftv, bothv;
    extern void  kgemv_(char *, integer *, integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *, 
	    integer *, halfcomplex *, halfcomplex *, integer *);
    logical somev;
    extern void  kcopy_(integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *), hlabad_(halfreal *, halfreal *);
    extern halfreal hlamch_(char *);
    extern void  xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern void  whscal_(integer *, halfreal *, 
	    halfcomplex *, integer *);
    extern integer ikamax_(integer *, halfcomplex *, integer *);
    extern void  klaset_(char *, integer *, integer *, 
	    halfcomplex *, halfcomplex *, halfcomplex *, integer *);
    logical rightv;
    extern halfreal hwasum_(integer *, halfcomplex *, integer *);
    extern void  klacpy_(char *, integer *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *);
    integer maxwrk;
    halfreal smlnum;
    extern void  klatrs_(char *, char *, char *, char *, 
	    integer *, halfcomplex *, integer *, halfcomplex *, 
	    halfreal *, halfreal *, integer *);
    logical lquery;


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */


/*     Decode and test the input parameters */

    /* Parameter adjustments */
    --select;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;
    --rwork;

    /* Function Body */
    bothv = lsame_(side, "B");
    rightv = lsame_(side, "R") || bothv;
    leftv = lsame_(side, "L") || bothv;

    allv = lsame_(howmny, "A");
    over = lsame_(howmny, "B");
    somev = lsame_(howmny, "S");

/*     Set M to the number of columns required to store the selected */
/*     eigenvectors. */

    if (somev) {
	*m = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (select[j]) {
		++(*m);
	    }
/* L10: */
	}
    } else {
	*m = *n;
    }

    *info = 0;
/* Writing concatenation */
    i__2[0] = 1, a__1[0] = side;
    i__2[1] = 1, a__1[1] = howmny;
    s_cat(ch__1, a__1, i__2, &c__2, (ftnlen)2);
    nb = ilaenv_(&c__1, "ZTREVC", ch__1, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)2);
    maxwrk = *n + (*n << 1) * nb;
    work[1].r = (halfreal) maxwrk, work[1].i = 0.;
    rwork[1] = (halfreal) (*n);
    lquery = *lwork == -1 || *lrwork == -1;
    if (! rightv && ! leftv) {
	*info = -1;
    } else if (! allv && ! over && ! somev) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ldt < f2cmax(1,*n)) {
	*info = -6;
    } else if (*ldvl < 1 || leftv && *ldvl < *n) {
	*info = -8;
    } else if (*ldvr < 1 || rightv && *ldvr < *n) {
	*info = -10;
    } else if (*mm < *m) {
	*info = -11;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = 1, i__3 = *n << 1;
	if (*lwork < f2cmax(i__1,i__3) && ! lquery) {
	    *info = -14;
	} else if (*lrwork < f2cmax(1,*n) && ! lquery) {
	    *info = -16;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTREVC3", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return;
    }

/*     Use blocked version of back-transformation if sufficient workspace. */
/*     Zero-out the workspace to avoid potential NaN propagation. */

    if (over && *lwork >= *n + (*n << 4)) {
	nb = (*lwork - *n) / (*n << 1);
	nb = f2cmin(nb,128);
	i__1 = (nb << 1) + 1;
	klaset_("F", n, &i__1, &c_b1, &c_b1, &work[1], n);
    } else {
	nb = 1;
    }

/*     Set the constants to control overflow. */

    unfl = hlamch_("Safe minimum");
    ovfl = 1. / unfl;
    hlabad_(&unfl, &ovfl);
    ulp = hlamch_("Precision");
    smlnum = unfl * (*n / ulp);

/*     Store the diagonal elements of T in working array WORK. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__3 = i__;
	i__4 = i__ + i__ * t_dim1;
	work[i__3].r = t[i__4].r, work[i__3].i = t[i__4].i;
/* L20: */
    }

/*     Compute 1-norm of each column of strictly upper triangular */
/*     part of T to control overflow in triangular solver. */

    rwork[1] = 0.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	i__3 = j - 1;
	rwork[j] = hwasum_(&i__3, &t[j * t_dim1 + 1], &c__1);
/* L30: */
    }

    if (rightv) {

/*        ============================================================ */
/*        Compute right eigenvectors. */

/*        IV is index of column in current block. */
/*        Non-blocked version always uses IV=NB=1; */
/*        blocked     version starts with IV=NB, goes down to 1. */
/*        (Note the "0-th" column is used to store the original diagonal.) */
	iv = nb;
	is = *m;
	for (ki = *n; ki >= 1; --ki) {
	    if (somev) {
		if (! select[ki]) {
		    goto L80;
		}
	    }
/* Computing MAX */
	    i__1 = ki + ki * t_dim1;
	    d__3 = ulp * ((d__1 = t[i__1].r, abs(d__1)) + (d__2 = d_imag(&t[
		    ki + ki * t_dim1]), abs(d__2)));
	    smin = f2cmax(d__3,smlnum);

/*           -------------------------------------------------------- */
/*           Complex right eigenvector */

	    i__1 = ki + iv * *n;
	    work[i__1].r = 1., work[i__1].i = 0.;

/*           Form right-hand side. */

	    i__1 = ki - 1;
	    for (k = 1; k <= i__1; ++k) {
		i__3 = k + iv * *n;
		i__4 = k + ki * t_dim1;
		z__1.r = -t[i__4].r, z__1.i = -t[i__4].i;
		work[i__3].r = z__1.r, work[i__3].i = z__1.i;
/* L40: */
	    }

/*           Solve upper triangular system: */
/*           [ T(1:KI-1,1:KI-1) - T(KI,KI) ]*X = SCALE*WORK. */

	    i__1 = ki - 1;
	    for (k = 1; k <= i__1; ++k) {
		i__3 = k + k * t_dim1;
		i__4 = k + k * t_dim1;
		i__5 = ki + ki * t_dim1;
		z__1.r = t[i__4].r - t[i__5].r, z__1.i = t[i__4].i - t[i__5]
			.i;
		t[i__3].r = z__1.r, t[i__3].i = z__1.i;
		i__3 = k + k * t_dim1;
		if ((d__1 = t[i__3].r, abs(d__1)) + (d__2 = d_imag(&t[k + k * 
			t_dim1]), abs(d__2)) < smin) {
		    i__4 = k + k * t_dim1;
		    t[i__4].r = smin, t[i__4].i = 0.;
		}
/* L50: */
	    }

	    if (ki > 1) {
		i__1 = ki - 1;
		klatrs_("Upper", "No transpose", "Non-unit", "Y", &i__1, &t[
			t_offset], ldt, &work[iv * *n + 1], &scale, &rwork[1],
			 info);
		i__1 = ki + iv * *n;
		work[i__1].r = scale, work[i__1].i = 0.;
	    }

/*           Copy the vector x or Q*x to VR and normalize. */

	    if (! over) {
/*              ------------------------------ */
/*              no back-transform: copy x to VR and normalize. */
		kcopy_(&ki, &work[iv * *n + 1], &c__1, &vr[is * vr_dim1 + 1], 
			&c__1);

		ii = ikamax_(&ki, &vr[is * vr_dim1 + 1], &c__1);
		i__1 = ii + is * vr_dim1;
		remax = 1. / ((d__1 = vr[i__1].r, abs(d__1)) + (d__2 = d_imag(
			&vr[ii + is * vr_dim1]), abs(d__2)));
		whscal_(&ki, &remax, &vr[is * vr_dim1 + 1], &c__1);

		i__1 = *n;
		for (k = ki + 1; k <= i__1; ++k) {
		    i__3 = k + is * vr_dim1;
		    vr[i__3].r = 0., vr[i__3].i = 0.;
/* L60: */
		}

	    } else if (nb == 1) {
/*              ------------------------------ */
/*              version 1: back-transform each vector with GEMV, Q*x. */
		if (ki > 1) {
		    i__1 = ki - 1;
		    z__1.r = scale, z__1.i = 0.;
		    kgemv_("N", n, &i__1, &c_b2, &vr[vr_offset], ldvr, &work[
			    iv * *n + 1], &c__1, &z__1, &vr[ki * vr_dim1 + 1],
			     &c__1);
		}

		ii = ikamax_(n, &vr[ki * vr_dim1 + 1], &c__1);
		i__1 = ii + ki * vr_dim1;
		remax = 1. / ((d__1 = vr[i__1].r, abs(d__1)) + (d__2 = d_imag(
			&vr[ii + ki * vr_dim1]), abs(d__2)));
		whscal_(n, &remax, &vr[ki * vr_dim1 + 1], &c__1);

	    } else {
/*              ------------------------------ */
/*              version 2: back-transform block of vectors with GEMM */
/*              zero out below vector */
		i__1 = *n;
		for (k = ki + 1; k <= i__1; ++k) {
		    i__3 = k + iv * *n;
		    work[i__3].r = 0., work[i__3].i = 0.;
		}

/*              Columns IV:NB of work are valid vectors. */
/*              When the number of vectors stored reaches NB, */
/*              or if this was last vector, do the GEMM */
		if (iv == 1 || ki == 1) {
		    i__1 = nb - iv + 1;
		    i__3 = ki + nb - iv;
		    kgemm_("N", "N", n, &i__1, &i__3, &c_b2, &vr[vr_offset], 
			    ldvr, &work[iv * *n + 1], n, &c_b1, &work[(nb + 
			    iv) * *n + 1], n);
/*                 normalize vectors */
		    i__1 = nb;
		    for (k = iv; k <= i__1; ++k) {
			ii = ikamax_(n, &work[(nb + k) * *n + 1], &c__1);
			i__3 = ii + (nb + k) * *n;
			remax = 1. / ((d__1 = work[i__3].r, abs(d__1)) + (
				d__2 = d_imag(&work[ii + (nb + k) * *n]), abs(
				d__2)));
			whscal_(n, &remax, &work[(nb + k) * *n + 1], &c__1);
		    }
		    i__1 = nb - iv + 1;
		    klacpy_("F", n, &i__1, &work[(nb + iv) * *n + 1], n, &vr[
			    ki * vr_dim1 + 1], ldvr);
		    iv = nb;
		} else {
		    --iv;
		}
	    }

/*           Restore the original diagonal elements of T. */

	    i__1 = ki - 1;
	    for (k = 1; k <= i__1; ++k) {
		i__3 = k + k * t_dim1;
		i__4 = k;
		t[i__3].r = work[i__4].r, t[i__3].i = work[i__4].i;
/* L70: */
	    }

	    --is;
L80:
	    ;
	}
    }

    if (leftv) {

/*        ============================================================ */
/*        Compute left eigenvectors. */

/*        IV is index of column in current block. */
/*        Non-blocked version always uses IV=1; */
/*        blocked     version starts with IV=1, goes up to NB. */
/*        (Note the "0-th" column is used to store the original diagonal.) */
	iv = 1;
	is = 1;
	i__1 = *n;
	for (ki = 1; ki <= i__1; ++ki) {

	    if (somev) {
		if (! select[ki]) {
		    goto L130;
		}
	    }
/* Computing MAX */
	    i__3 = ki + ki * t_dim1;
	    d__3 = ulp * ((d__1 = t[i__3].r, abs(d__1)) + (d__2 = d_imag(&t[
		    ki + ki * t_dim1]), abs(d__2)));
	    smin = f2cmax(d__3,smlnum);

/*           -------------------------------------------------------- */
/*           Complex left eigenvector */

	    i__3 = ki + iv * *n;
	    work[i__3].r = 1., work[i__3].i = 0.;

/*           Form right-hand side. */

	    i__3 = *n;
	    for (k = ki + 1; k <= i__3; ++k) {
		i__4 = k + iv * *n;
		d_cnjg(&z__2, &t[ki + k * t_dim1]);
		z__1.r = -z__2.r, z__1.i = -z__2.i;
		work[i__4].r = z__1.r, work[i__4].i = z__1.i;
/* L90: */
	    }

/*           Solve conjugate-transposed triangular system: */
/*           [ T(KI+1:N,KI+1:N) - T(KI,KI) ]**H * X = SCALE*WORK. */

	    i__3 = *n;
	    for (k = ki + 1; k <= i__3; ++k) {
		i__4 = k + k * t_dim1;
		i__5 = k + k * t_dim1;
		i__6 = ki + ki * t_dim1;
		z__1.r = t[i__5].r - t[i__6].r, z__1.i = t[i__5].i - t[i__6]
			.i;
		t[i__4].r = z__1.r, t[i__4].i = z__1.i;
		i__4 = k + k * t_dim1;
		if ((d__1 = t[i__4].r, abs(d__1)) + (d__2 = d_imag(&t[k + k * 
			t_dim1]), abs(d__2)) < smin) {
		    i__5 = k + k * t_dim1;
		    t[i__5].r = smin, t[i__5].i = 0.;
		}
/* L100: */
	    }

	    if (ki < *n) {
		i__3 = *n - ki;
		klatrs_("Upper", "Conjugate transpose", "Non-unit", "Y", &
			i__3, &t[ki + 1 + (ki + 1) * t_dim1], ldt, &work[ki + 
			1 + iv * *n], &scale, &rwork[1], info);
		i__3 = ki + iv * *n;
		work[i__3].r = scale, work[i__3].i = 0.;
	    }

/*           Copy the vector x or Q*x to VL and normalize. */

	    if (! over) {
/*              ------------------------------ */
/*              no back-transform: copy x to VL and normalize. */
		i__3 = *n - ki + 1;
		kcopy_(&i__3, &work[ki + iv * *n], &c__1, &vl[ki + is * 
			vl_dim1], &c__1);

		i__3 = *n - ki + 1;
		ii = ikamax_(&i__3, &vl[ki + is * vl_dim1], &c__1) + ki - 1;
		i__3 = ii + is * vl_dim1;
		remax = 1. / ((d__1 = vl[i__3].r, abs(d__1)) + (d__2 = d_imag(
			&vl[ii + is * vl_dim1]), abs(d__2)));
		i__3 = *n - ki + 1;
		whscal_(&i__3, &remax, &vl[ki + is * vl_dim1], &c__1);

		i__3 = ki - 1;
		for (k = 1; k <= i__3; ++k) {
		    i__4 = k + is * vl_dim1;
		    vl[i__4].r = 0., vl[i__4].i = 0.;
/* L110: */
		}

	    } else if (nb == 1) {
/*              ------------------------------ */
/*              version 1: back-transform each vector with GEMV, Q*x. */
		if (ki < *n) {
		    i__3 = *n - ki;
		    z__1.r = scale, z__1.i = 0.;
		    kgemv_("N", n, &i__3, &c_b2, &vl[(ki + 1) * vl_dim1 + 1], 
			    ldvl, &work[ki + 1 + iv * *n], &c__1, &z__1, &vl[
			    ki * vl_dim1 + 1], &c__1);
		}

		ii = ikamax_(n, &vl[ki * vl_dim1 + 1], &c__1);
		i__3 = ii + ki * vl_dim1;
		remax = 1. / ((d__1 = vl[i__3].r, abs(d__1)) + (d__2 = d_imag(
			&vl[ii + ki * vl_dim1]), abs(d__2)));
		whscal_(n, &remax, &vl[ki * vl_dim1 + 1], &c__1);

	    } else {
/*              ------------------------------ */
/*              version 2: back-transform block of vectors with GEMM */
/*              zero out above vector */
/*              could go from KI-NV+1 to KI-1 */
		i__3 = ki - 1;
		for (k = 1; k <= i__3; ++k) {
		    i__4 = k + iv * *n;
		    work[i__4].r = 0., work[i__4].i = 0.;
		}

/*              Columns 1:IV of work are valid vectors. */
/*              When the number of vectors stored reaches NB, */
/*              or if this was last vector, do the GEMM */
		if (iv == nb || ki == *n) {
		    i__3 = *n - ki + iv;
		    kgemm_("N", "N", n, &iv, &i__3, &c_b2, &vl[(ki - iv + 1) *
			     vl_dim1 + 1], ldvl, &work[ki - iv + 1 + *n], n, &
			    c_b1, &work[(nb + 1) * *n + 1], n);
/*                 normalize vectors */
		    i__3 = iv;
		    for (k = 1; k <= i__3; ++k) {
			ii = ikamax_(n, &work[(nb + k) * *n + 1], &c__1);
			i__4 = ii + (nb + k) * *n;
			remax = 1. / ((d__1 = work[i__4].r, abs(d__1)) + (
				d__2 = d_imag(&work[ii + (nb + k) * *n]), abs(
				d__2)));
			whscal_(n, &remax, &work[(nb + k) * *n + 1], &c__1);
		    }
		    klacpy_("F", n, &iv, &work[(nb + 1) * *n + 1], n, &vl[(ki 
			    - iv + 1) * vl_dim1 + 1], ldvl);
		    iv = 1;
		} else {
		    ++iv;
		}
	    }

/*           Restore the original diagonal elements of T. */

	    i__3 = *n;
	    for (k = ki + 1; k <= i__3; ++k) {
		i__4 = k + k * t_dim1;
		i__5 = k;
		t[i__4].r = work[i__5].r, t[i__4].i = work[i__5].i;
/* L120: */
	    }

	    ++is;
L130:
	    ;
	}
    }

    return;

/*     End of ZTREVC3 */

} /* ktrevc3_ */

