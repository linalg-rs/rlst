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
static quadcomplex c_b2 = {1.,0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static logical c_true = TRUE_;
static integer c__12 = 12;

/* > \brief \b ZLAQR3 performs the unitary similarity transformation of a Hessenberg matrix to detect and defl
ate fully converged eigenvalues from a trailing principal submatrix (aggressive early deflation). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLAQR3 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlaqr3.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlaqr3.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlaqr3.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLAQR3( WANTT, WANTZ, N, KTOP, KBOT, NW, H, LDH, ILOZ, */
/*                          IHIZ, Z, LDZ, NS, ND, SH, V, LDV, NH, T, LDT, */
/*                          NV, WV, LDWV, WORK, LWORK ) */

/*       INTEGER            IHIZ, ILOZ, KBOT, KTOP, LDH, LDT, LDV, LDWV, */
/*      $                   LDZ, LWORK, N, ND, NH, NS, NV, NW */
/*       LOGICAL            WANTT, WANTZ */
/*       COMPLEX*16         H( LDH, * ), SH( * ), T( LDT, * ), V( LDV, * ), */
/*      $                   WORK( * ), WV( LDWV, * ), Z( LDZ, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >    Aggressive early deflation: */
/* > */
/* >    ZLAQR3 accepts as input an upper Hessenberg matrix */
/* >    H and performs an unitary similarity transformation */
/* >    designed to detect and deflate fully converged eigenvalues from */
/* >    a trailing principal submatrix.  On output H has been over- */
/* >    written by a new Hessenberg matrix that is a perturbation of */
/* >    an unitary similarity transformation of H.  It is to be */
/* >    hoped that the final version of H has many zero subdiagonal */
/* >    entries. */
/* > */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] WANTT */
/* > \verbatim */
/* >          WANTT is LOGICAL */
/* >          If .TRUE., then the Hessenberg matrix H is fully updated */
/* >          so that the triangular Schur factor may be */
/* >          computed (in cooperation with the calling subroutine). */
/* >          If .FALSE., then only enough of H is updated to preserve */
/* >          the eigenvalues. */
/* > \endverbatim */
/* > */
/* > \param[in] WANTZ */
/* > \verbatim */
/* >          WANTZ is LOGICAL */
/* >          If .TRUE., then the unitary matrix Z is updated so */
/* >          so that the unitary Schur factor may be computed */
/* >          (in cooperation with the calling subroutine). */
/* >          If .FALSE., then Z is not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix H and (if WANTZ is .TRUE.) the */
/* >          order of the unitary matrix Z. */
/* > \endverbatim */
/* > */
/* > \param[in] KTOP */
/* > \verbatim */
/* >          KTOP is INTEGER */
/* >          It is assumed that either KTOP = 1 or H(KTOP,KTOP-1)=0. */
/* >          KBOT and KTOP together determine an isolated block */
/* >          along the diagonal of the Hessenberg matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] KBOT */
/* > \verbatim */
/* >          KBOT is INTEGER */
/* >          It is assumed without a check that either */
/* >          KBOT = N or H(KBOT+1,KBOT)=0.  KBOT and KTOP together */
/* >          determine an isolated block along the diagonal of the */
/* >          Hessenberg matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] NW */
/* > \verbatim */
/* >          NW is INTEGER */
/* >          Deflation window size.  1 .LE. NW .LE. (KBOT-KTOP+1). */
/* > \endverbatim */
/* > */
/* > \param[in,out] H */
/* > \verbatim */
/* >          H is COMPLEX*16 array, dimension (LDH,N) */
/* >          On input the initial N-by-N section of H stores the */
/* >          Hessenberg matrix undergoing aggressive early deflation. */
/* >          On output H has been transformed by a unitary */
/* >          similarity transformation, perturbed, and the returned */
/* >          to Hessenberg form that (it is to be hoped) has some */
/* >          zero subdiagonal entries. */
/* > \endverbatim */
/* > */
/* > \param[in] LDH */
/* > \verbatim */
/* >          LDH is INTEGER */
/* >          Leading dimension of H just as declared in the calling */
/* >          subroutine.  N .LE. LDH */
/* > \endverbatim */
/* > */
/* > \param[in] ILOZ */
/* > \verbatim */
/* >          ILOZ is INTEGER */
/* > \endverbatim */
/* > */
/* > \param[in] IHIZ */
/* > \verbatim */
/* >          IHIZ is INTEGER */
/* >          Specify the rows of Z to which transformations must be */
/* >          applied if WANTZ is .TRUE.. 1 .LE. ILOZ .LE. IHIZ .LE. N. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Z */
/* > \verbatim */
/* >          Z is COMPLEX*16 array, dimension (LDZ,N) */
/* >          IF WANTZ is .TRUE., then on output, the unitary */
/* >          similarity transformation mentioned above has been */
/* >          accumulated into Z(ILOZ:IHIZ,ILOZ:IHIZ) from the right. */
/* >          If WANTZ is .FALSE., then Z is unreferenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDZ */
/* > \verbatim */
/* >          LDZ is INTEGER */
/* >          The leading dimension of Z just as declared in the */
/* >          calling subroutine.  1 .LE. LDZ. */
/* > \endverbatim */
/* > */
/* > \param[out] NS */
/* > \verbatim */
/* >          NS is INTEGER */
/* >          The number of unconverged (ie approximate) eigenvalues */
/* >          returned in SR and SI that may be used as shifts by the */
/* >          calling subroutine. */
/* > \endverbatim */
/* > */
/* > \param[out] ND */
/* > \verbatim */
/* >          ND is INTEGER */
/* >          The number of converged eigenvalues uncovered by this */
/* >          subroutine. */
/* > \endverbatim */
/* > */
/* > \param[out] SH */
/* > \verbatim */
/* >          SH is COMPLEX*16 array, dimension (KBOT) */
/* >          On output, approximate eigenvalues that may */
/* >          be used for shifts are stored in SH(KBOT-ND-NS+1) */
/* >          through SR(KBOT-ND).  Converged eigenvalues are */
/* >          stored in SH(KBOT-ND+1) through SH(KBOT). */
/* > \endverbatim */
/* > */
/* > \param[out] V */
/* > \verbatim */
/* >          V is COMPLEX*16 array, dimension (LDV,NW) */
/* >          An NW-by-NW work array. */
/* > \endverbatim */
/* > */
/* > \param[in] LDV */
/* > \verbatim */
/* >          LDV is INTEGER */
/* >          The leading dimension of V just as declared in the */
/* >          calling subroutine.  NW .LE. LDV */
/* > \endverbatim */
/* > */
/* > \param[in] NH */
/* > \verbatim */
/* >          NH is INTEGER */
/* >          The number of columns of T.  NH.GE.NW. */
/* > \endverbatim */
/* > */
/* > \param[out] T */
/* > \verbatim */
/* >          T is COMPLEX*16 array, dimension (LDT,NW) */
/* > \endverbatim */
/* > */
/* > \param[in] LDT */
/* > \verbatim */
/* >          LDT is INTEGER */
/* >          The leading dimension of T just as declared in the */
/* >          calling subroutine.  NW .LE. LDT */
/* > \endverbatim */
/* > */
/* > \param[in] NV */
/* > \verbatim */
/* >          NV is INTEGER */
/* >          The number of rows of work array WV available for */
/* >          workspace.  NV.GE.NW. */
/* > \endverbatim */
/* > */
/* > \param[out] WV */
/* > \verbatim */
/* >          WV is COMPLEX*16 array, dimension (LDWV,NW) */
/* > \endverbatim */
/* > */
/* > \param[in] LDWV */
/* > \verbatim */
/* >          LDWV is INTEGER */
/* >          The leading dimension of W just as declared in the */
/* >          calling subroutine.  NW .LE. LDV */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (LWORK) */
/* >          On exit, WORK(1) is set to an estimate of the optimal value */
/* >          of LWORK for the given values of N, NW, KTOP and KBOT. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the work array WORK.  LWORK = 2*NW */
/* >          suffices, but greater efficiency may result from larger */
/* >          values of LWORK. */
/* > */
/* >          If LWORK = -1, then a workspace query is assumed; ZLAQR3 */
/* >          only estimates the optimal workspace size for the given */
/* >          values of N, NW, KTOP and KBOT.  The estimate is returned */
/* >          in WORK(1).  No error message related to LWORK is issued */
/* >          by XERBLA.  Neither H nor Z are accessed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complex16OTHERauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >       Karen Braman and Ralph Byers, Department of Mathematics, */
/* >       University of Kansas, USA */
/* > */
/*  ===================================================================== */
void  wlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, quadcomplex *h__, 
	integer *ldh, integer *iloz, integer *ihiz, quadcomplex *z__, 
	integer *ldz, integer *ns, integer *nd, quadcomplex *sh, 
	quadcomplex *v, integer *ldv, integer *nh, quadcomplex *t, 
	integer *ldt, integer *nv, quadcomplex *wv, integer *ldwv, 
	quadcomplex *work, integer *lwork)
{
    /* System generated locals */
    integer h_dim1, h_offset, t_dim1, t_offset, v_dim1, v_offset, wv_dim1, 
	    wv_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4;
    quadreal d__1, d__2, d__3, d__4, d__5, d__6;
    quadcomplex z__1, z__2;

    /* Local variables */
    integer i__, j;
    quadcomplex s;
    integer jw;
    quadreal foo;
    integer kln;
    quadcomplex tau;
    integer knt;
    quadreal ulp;
    integer lwk1, lwk2, lwk3;
    quadcomplex beta;
    integer kcol, info, nmin, ifst, ilst, ltop, krow;
    extern void  wlarf_(char *, integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, quadcomplex *, 
	    integer *, quadcomplex *);
    integer infqr;
    extern void  wgemm_(char *, char *, integer *, integer *, 
	    integer *, quadcomplex *, quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, quadcomplex *, 
	    integer *);
    integer kwtop;
    extern void  wcopy_(integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *), qlabad_(quadreal *, quadreal *), 
	    wlaqr4_(logical *, logical *, integer *, integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, integer *,
	     quadcomplex *, integer *, quadcomplex *, integer *, integer *
	    );
    extern quadreal qlamch_(char *);
    quadreal safmin;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    quadreal safmax;
    extern void  wgehrd_(integer *, integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, quadcomplex *, 
	    integer *, integer *), wlarfg_(integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *), wlahqr_(logical *, 
	    logical *, integer *, integer *, integer *, quadcomplex *, 
	    integer *, quadcomplex *, integer *, integer *, quadcomplex *,
	     integer *, integer *), wlacpy_(char *, integer *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *), 
	    wlaset_(char *, integer *, integer *, quadcomplex *, 
	    quadcomplex *, quadcomplex *, integer *);
    quadreal smlnum;
    extern void  wtrexc_(char *, integer *, quadcomplex *, 
	    integer *, quadcomplex *, integer *, integer *, integer *, 
	    integer *);
    integer lwkopt;
    extern void  wunmhr_(char *, char *, integer *, integer *, 
	    integer *, integer *, quadcomplex *, integer *, quadcomplex *,
	     quadcomplex *, integer *, quadcomplex *, integer *, integer *
	    );


/*  -- LAPACK auxiliary routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ================================================================ */


/*     ==== Estimate optimal workspace. ==== */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --sh;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    --work;

    /* Function Body */
/* Computing MIN */
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = f2cmin(i__1,i__2);
    if (jw <= 2) {
	lwkopt = 1;
    } else {

/*        ==== Workspace query call to ZGEHRD ==== */

	i__1 = jw - 1;
	wgehrd_(&jw, &c__1, &i__1, &t[t_offset], ldt, &work[1], &work[1], &
		c_n1, &info);
	lwk1 = (integer) work[1].r;

/*        ==== Workspace query call to ZUNMHR ==== */

	i__1 = jw - 1;
	wunmhr_("R", "N", &jw, &jw, &c__1, &i__1, &t[t_offset], ldt, &work[1],
		 &v[v_offset], ldv, &work[1], &c_n1, &info);
	lwk2 = (integer) work[1].r;

/*        ==== Workspace query call to ZLAQR4 ==== */

	wlaqr4_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sh[1], 
		&c__1, &jw, &v[v_offset], ldv, &work[1], &c_n1, &infqr);
	lwk3 = (integer) work[1].r;

/*        ==== Optimal workspace ==== */

/* Computing MAX */
	i__1 = jw + f2cmax(lwk1,lwk2);
	lwkopt = f2cmax(i__1,lwk3);
    }

/*     ==== Quick return in case of workspace query. ==== */

    if (*lwork == -1) {
	d__1 = (quadreal) lwkopt;
	z__1.r = d__1, z__1.i = 0.;
	work[1].r = z__1.r, work[1].i = z__1.i;
	return;
    }

/*     ==== Nothing to do ... */
/*     ... for an empty active block ... ==== */
    *ns = 0;
    *nd = 0;
    work[1].r = 1., work[1].i = 0.;
    if (*ktop > *kbot) {
	return;
    }
/*     ... nor for an empty deflation window. ==== */
    if (*nw < 1) {
	return;
    }

/*     ==== Machine constants ==== */

    safmin = qlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    qlabad_(&safmin, &safmax);
    ulp = qlamch_("PRECISION");
    smlnum = safmin * ((quadreal) (*n) / ulp);

/*     ==== Setup deflation window ==== */

/* Computing MIN */
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = f2cmin(i__1,i__2);
    kwtop = *kbot - jw + 1;
    if (kwtop == *ktop) {
	s.r = 0., s.i = 0.;
    } else {
	i__1 = kwtop + (kwtop - 1) * h_dim1;
	s.r = h__[i__1].r, s.i = h__[i__1].i;
    }

    if (*kbot == kwtop) {

/*        ==== 1-by-1 deflation window: not much to do ==== */

	i__1 = kwtop;
	i__2 = kwtop + kwtop * h_dim1;
	sh[i__1].r = h__[i__2].r, sh[i__1].i = h__[i__2].i;
	*ns = 1;
	*nd = 0;
/* Computing MAX */
	i__1 = kwtop + kwtop * h_dim1;
	d__5 = smlnum, d__6 = ulp * ((d__1 = h__[i__1].r, abs(d__1)) + (d__2 =
		 d_imag(&h__[kwtop + kwtop * h_dim1]), abs(d__2)));
	if ((d__3 = s.r, abs(d__3)) + (d__4 = d_imag(&s), abs(d__4)) <= f2cmax(
		d__5,d__6)) {
	    *ns = 0;
	    *nd = 1;
	    if (kwtop > *ktop) {
		i__1 = kwtop + (kwtop - 1) * h_dim1;
		h__[i__1].r = 0., h__[i__1].i = 0.;
	    }
	}
	work[1].r = 1., work[1].i = 0.;
	return;
    }

/*     ==== Convert to spike-triangular form.  (In case of a */
/*     .    rare QR failure, this routine continues to do */
/*     .    aggressive early deflation using that part of */
/*     .    the deflation window that converged using INFQR */
/*     .    here and there to keep track.) ==== */

    wlacpy_("U", &jw, &jw, &h__[kwtop + kwtop * h_dim1], ldh, &t[t_offset], 
	    ldt);
    i__1 = jw - 1;
    i__2 = *ldh + 1;
    i__3 = *ldt + 1;
    wcopy_(&i__1, &h__[kwtop + 1 + kwtop * h_dim1], &i__2, &t[t_dim1 + 2], &
	    i__3);

    wlaset_("A", &jw, &jw, &c_b1, &c_b2, &v[v_offset], ldv);
    nmin = ilaenv_(&c__12, "ZLAQR3", "SV", &jw, &c__1, &jw, lwork, (ftnlen)6, 
	    (ftnlen)2);
    if (jw > nmin) {
	wlaqr4_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sh[
		kwtop], &c__1, &jw, &v[v_offset], ldv, &work[1], lwork, &
		infqr);
    } else {
	wlahqr_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sh[
		kwtop], &c__1, &jw, &v[v_offset], ldv, &infqr);
    }

/*     ==== Deflation detection loop ==== */

    *ns = jw;
    ilst = infqr + 1;
    i__1 = jw;
    for (knt = infqr + 1; knt <= i__1; ++knt) {

/*        ==== Small spike tip deflation test ==== */

	i__2 = *ns + *ns * t_dim1;
	foo = (d__1 = t[i__2].r, abs(d__1)) + (d__2 = d_imag(&t[*ns + *ns * 
		t_dim1]), abs(d__2));
	if (foo == 0.) {
	    foo = (d__1 = s.r, abs(d__1)) + (d__2 = d_imag(&s), abs(d__2));
	}
	i__2 = *ns * v_dim1 + 1;
/* Computing MAX */
	d__5 = smlnum, d__6 = ulp * foo;
	if (((d__1 = s.r, abs(d__1)) + (d__2 = d_imag(&s), abs(d__2))) * ((
		d__3 = v[i__2].r, abs(d__3)) + (d__4 = d_imag(&v[*ns * v_dim1 
		+ 1]), abs(d__4))) <= f2cmax(d__5,d__6)) {

/*           ==== One more converged eigenvalue ==== */

	    --(*ns);
	} else {

/*           ==== One undeflatable eigenvalue.  Move it up out of the */
/*           .    way.   (ZTREXC can not fail in this case.) ==== */

	    ifst = *ns;
	    wtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst, &
		    ilst, &info);
	    ++ilst;
	}
/* L10: */
    }

/*        ==== Return to Hessenberg form ==== */

    if (*ns == 0) {
	s.r = 0., s.i = 0.;
    }

    if (*ns < jw) {

/*        ==== sorting the diagonal of T improves accuracy for */
/*        .    graded matrices.  ==== */

	i__1 = *ns;
	for (i__ = infqr + 1; i__ <= i__1; ++i__) {
	    ifst = i__;
	    i__2 = *ns;
	    for (j = i__ + 1; j <= i__2; ++j) {
		i__3 = j + j * t_dim1;
		i__4 = ifst + ifst * t_dim1;
		if ((d__1 = t[i__3].r, abs(d__1)) + (d__2 = d_imag(&t[j + j * 
			t_dim1]), abs(d__2)) > (d__3 = t[i__4].r, abs(d__3)) 
			+ (d__4 = d_imag(&t[ifst + ifst * t_dim1]), abs(d__4))
			) {
		    ifst = j;
		}
/* L20: */
	    }
	    ilst = i__;
	    if (ifst != ilst) {
		wtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst,
			 &ilst, &info);
	    }
/* L30: */
	}
    }

/*     ==== Restore shift/eigenvalue array from T ==== */

    i__1 = jw;
    for (i__ = infqr + 1; i__ <= i__1; ++i__) {
	i__2 = kwtop + i__ - 1;
	i__3 = i__ + i__ * t_dim1;
	sh[i__2].r = t[i__3].r, sh[i__2].i = t[i__3].i;
/* L40: */
    }


    if (*ns < jw || s.r == 0. && s.i == 0.) {
	if (*ns > 1 && (s.r != 0. || s.i != 0.)) {

/*           ==== Reflect spike back into lower triangle ==== */

	    wcopy_(ns, &v[v_offset], ldv, &work[1], &c__1);
	    i__1 = *ns;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		d_cnjg(&z__1, &work[i__]);
		work[i__2].r = z__1.r, work[i__2].i = z__1.i;
/* L50: */
	    }
	    beta.r = work[1].r, beta.i = work[1].i;
	    wlarfg_(ns, &beta, &work[2], &c__1, &tau);
	    work[1].r = 1., work[1].i = 0.;

	    i__1 = jw - 2;
	    i__2 = jw - 2;
	    wlaset_("L", &i__1, &i__2, &c_b1, &c_b1, &t[t_dim1 + 3], ldt);

	    d_cnjg(&z__1, &tau);
	    wlarf_("L", ns, &jw, &work[1], &c__1, &z__1, &t[t_offset], ldt, &
		    work[jw + 1]);
	    wlarf_("R", ns, ns, &work[1], &c__1, &tau, &t[t_offset], ldt, &
		    work[jw + 1]);
	    wlarf_("R", &jw, ns, &work[1], &c__1, &tau, &v[v_offset], ldv, &
		    work[jw + 1]);

	    i__1 = *lwork - jw;
	    wgehrd_(&jw, &c__1, ns, &t[t_offset], ldt, &work[1], &work[jw + 1]
		    , &i__1, &info);
	}

/*        ==== Copy updated reduced window into place ==== */

	if (kwtop > 1) {
	    i__1 = kwtop + (kwtop - 1) * h_dim1;
	    d_cnjg(&z__2, &v[v_dim1 + 1]);
	    z__1.r = s.r * z__2.r - s.i * z__2.i, z__1.i = s.r * z__2.i + s.i 
		    * z__2.r;
	    h__[i__1].r = z__1.r, h__[i__1].i = z__1.i;
	}
	wlacpy_("U", &jw, &jw, &t[t_offset], ldt, &h__[kwtop + kwtop * h_dim1]
		, ldh);
	i__1 = jw - 1;
	i__2 = *ldt + 1;
	i__3 = *ldh + 1;
	wcopy_(&i__1, &t[t_dim1 + 2], &i__2, &h__[kwtop + 1 + kwtop * h_dim1],
		 &i__3);

/*        ==== Accumulate orthogonal matrix in order update */
/*        .    H and Z, if requested.  ==== */

	if (*ns > 1 && (s.r != 0. || s.i != 0.)) {
	    i__1 = *lwork - jw;
	    wunmhr_("R", "N", &jw, ns, &c__1, ns, &t[t_offset], ldt, &work[1],
		     &v[v_offset], ldv, &work[jw + 1], &i__1, &info);
	}

/*        ==== Update vertical slab in H ==== */

	if (*wantt) {
	    ltop = 1;
	} else {
	    ltop = *ktop;
	}
	i__1 = kwtop - 1;
	i__2 = *nv;
	for (krow = ltop; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow += 
		i__2) {
/* Computing MIN */
	    i__3 = *nv, i__4 = kwtop - krow;
	    kln = f2cmin(i__3,i__4);
	    wgemm_("N", "N", &kln, &jw, &jw, &c_b2, &h__[krow + kwtop * 
		    h_dim1], ldh, &v[v_offset], ldv, &c_b1, &wv[wv_offset], 
		    ldwv);
	    wlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &h__[krow + kwtop * 
		    h_dim1], ldh);
/* L60: */
	}

/*        ==== Update horizontal slab in H ==== */

	if (*wantt) {
	    i__2 = *n;
	    i__1 = *nh;
	    for (kcol = *kbot + 1; i__1 < 0 ? kcol >= i__2 : kcol <= i__2; 
		    kcol += i__1) {
/* Computing MIN */
		i__3 = *nh, i__4 = *n - kcol + 1;
		kln = f2cmin(i__3,i__4);
		wgemm_("C", "N", &jw, &kln, &jw, &c_b2, &v[v_offset], ldv, &
			h__[kwtop + kcol * h_dim1], ldh, &c_b1, &t[t_offset], 
			ldt);
		wlacpy_("A", &jw, &kln, &t[t_offset], ldt, &h__[kwtop + kcol *
			 h_dim1], ldh);
/* L70: */
	    }
	}

/*        ==== Update vertical slab in Z ==== */

	if (*wantz) {
	    i__1 = *ihiz;
	    i__2 = *nv;
	    for (krow = *iloz; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow +=
		     i__2) {
/* Computing MIN */
		i__3 = *nv, i__4 = *ihiz - krow + 1;
		kln = f2cmin(i__3,i__4);
		wgemm_("N", "N", &kln, &jw, &jw, &c_b2, &z__[krow + kwtop * 
			z_dim1], ldz, &v[v_offset], ldv, &c_b1, &wv[wv_offset]
			, ldwv);
		wlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &z__[krow + 
			kwtop * z_dim1], ldz);
/* L80: */
	    }
	}
    }

/*     ==== Return the number of deflations ... ==== */

    *nd = jw - *ns;

/*     ==== ... and the number of shifts. (Subtracting */
/*     .    INFQR from the spike length takes care */
/*     .    of the case of a rare QR failure while */
/*     .    calculating eigenvalues of the deflation */
/*     .    window.)  ==== */

    *ns -= infqr;

/*      ==== Return optimal workspace. ==== */

    d__1 = (quadreal) lwkopt;
    z__1.r = d__1, z__1.i = 0.;
    work[1].r = z__1.r, work[1].i = z__1.i;

/*     ==== End of ZLAQR3 ==== */

    return;
} /* wlaqr3_ */

