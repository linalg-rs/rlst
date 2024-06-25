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

static halfcomplex c_b1 = {1.,0.};

/* > \brief \b ZGELQT3 recursively computes a LQ factorization of a general doublereal or complex matrix using the c
ompact WY representation of Q. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DGEQRT3 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgelqt3
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgelqt3
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgelqt3
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*        SUBROUTINE ZGELQT3( M, N, A, LDA, T, LDT, INFO ) */

/*       INTEGER   INFO, LDA, M, N, LDT */
/*       COMPLEX*16   A( LDA, * ), T( LDT, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DGELQT3 recursively computes a LQ factorization of a complex M-by-N */
/* > matrix A, using the compact WY representation of Q. */
/* > */
/* > Based on the algorithm of Elmroth and Gustavson, */
/* > IBM J. Res. Develop. Vol 44 No. 4 July 2000. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M =< N. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the doublereal M-by-N matrix A.  On exit, the elements on and */
/* >          below the diagonal contain the N-by-N lower triangular matrix L; the */
/* >          elements above the diagonal are the rows of V.  See below for */
/* >          further details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] T */
/* > \verbatim */
/* >          T is COMPLEX*16 array, dimension (LDT,N) */
/* >          The N-by-N upper triangular factor of the block reflector. */
/* >          The elements on and above the diagonal contain the block */
/* >          reflector T; the elements below the diagonal are not used. */
/* >          See below for further details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDT */
/* > \verbatim */
/* >          LDT is INTEGER */
/* >          The leading dimension of the array T.  LDT >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2017 */

/* > \ingroup doubleGEcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The matrix V stores the elementary reflectors H(i) in the i-th row */
/* >  above the diagonal. For example, if M=5 and N=3, the matrix V is */
/* > */
/* >               V = (  1  v1 v1 v1 v1 ) */
/* >                   (     1  v2 v2 v2 ) */
/* >                   (     1  v3 v3 v3 ) */
/* > */
/* > */
/* >  where the vi's represent the vectors which define H(i), which are returned */
/* >  in the matrix A.  The 1's along the diagonal of V are not stored in A.  The */
/* >  block reflector H is then given by */
/* > */
/* >               H = I - V * T * V**T */
/* > */
/* >  where V**T is the transpose of V. */
/* > */
/* >  For details of the algorithm, see Elmroth and Gustavson (cited above). */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  kgelqt3_(integer *m, integer *n, halfcomplex *a, 
	integer *lda, halfcomplex *t, integer *ldt, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, t_dim1, t_offset, i__1, i__2, i__3, i__4, i__5;
    halfcomplex z__1;

    /* Local variables */
    integer i__, j, i1, j1, m1, m2, iinfo;
    extern void  kgemm_(char *, char *, integer *, integer *, 
	    integer *, halfcomplex *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, halfcomplex *, 
	    integer *), ktrmm_(char *, char *, char *, char *,
	     integer *, integer *, halfcomplex *, halfcomplex *, integer *
	    , halfcomplex *, integer *), 
	    xerbla_(char *, integer *), klarfg_(integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *);


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -4;
    } else if (*ldt < f2cmax(1,*m)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGELQT3", &i__1);
	return;
    }

    if (*m == 1) {

/*        Compute Householder transform when N=1 */

	klarfg_(n, &a[a_offset], &a[f2cmin(2,*n) * a_dim1 + 1], lda, &t[t_offset]
		);
	i__1 = t_dim1 + 1;
	d_cnjg(&z__1, &t[t_dim1 + 1]);
	t[i__1].r = z__1.r, t[i__1].i = z__1.i;

    } else {

/*        Otherwise, split A into blocks... */

	m1 = *m / 2;
	m2 = *m - m1;
/* Computing MIN */
	i__1 = m1 + 1;
	i1 = f2cmin(i__1,*m);
/* Computing MIN */
	i__1 = *m + 1;
	j1 = f2cmin(i__1,*n);

/*        Compute A(1:M1,1:N) <- (Y1,R1,T1), where Q1 = I - Y1 T1 Y1^H */

	kgelqt3_(&m1, n, &a[a_offset], lda, &t[t_offset], ldt, &iinfo);

/*        Compute A(J1:M,1:N) =  A(J1:M,1:N) Q1^H [workspace: T(1:N1,J1:N)] */

	i__1 = m2;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = m1;
	    for (j = 1; j <= i__2; ++j) {
		i__3 = i__ + m1 + j * t_dim1;
		i__4 = i__ + m1 + j * a_dim1;
		t[i__3].r = a[i__4].r, t[i__3].i = a[i__4].i;
	    }
	}
	ktrmm_("R", "U", "C", "U", &m2, &m1, &c_b1, &a[a_offset], lda, &t[i1 
		+ t_dim1], ldt);

	i__1 = *n - m1;
	kgemm_("N", "C", &m2, &m1, &i__1, &c_b1, &a[i1 + i1 * a_dim1], lda, &
		a[i1 * a_dim1 + 1], lda, &c_b1, &t[i1 + t_dim1], ldt);

	ktrmm_("R", "U", "N", "N", &m2, &m1, &c_b1, &t[t_offset], ldt, &t[i1 
		+ t_dim1], ldt);

	i__1 = *n - m1;
	z__1.r = -1., z__1.i = -0.;
	kgemm_("N", "N", &m2, &i__1, &m1, &z__1, &t[i1 + t_dim1], ldt, &a[i1 *
		 a_dim1 + 1], lda, &c_b1, &a[i1 + i1 * a_dim1], lda);

	ktrmm_("R", "U", "N", "U", &m2, &m1, &c_b1, &a[a_offset], lda, &t[i1 
		+ t_dim1], ldt);

	i__1 = m2;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = m1;
	    for (j = 1; j <= i__2; ++j) {
		i__3 = i__ + m1 + j * a_dim1;
		i__4 = i__ + m1 + j * a_dim1;
		i__5 = i__ + m1 + j * t_dim1;
		z__1.r = a[i__4].r - t[i__5].r, z__1.i = a[i__4].i - t[i__5]
			.i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
		i__3 = i__ + m1 + j * t_dim1;
		t[i__3].r = 0., t[i__3].i = 0.;
	    }
	}

/*        Compute A(J1:M,J1:N) <- (Y2,R2,T2) where Q2 = I - Y2 T2 Y2^H */

	i__1 = *n - m1;
	kgelqt3_(&m2, &i__1, &a[i1 + i1 * a_dim1], lda, &t[i1 + i1 * t_dim1], 
		ldt, &iinfo);

/*        Compute T3 = T(J1:N1,1:N) = -T1 Y1^H Y2 T2 */

	i__1 = m2;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = m1;
	    for (j = 1; j <= i__2; ++j) {
		i__3 = j + (i__ + m1) * t_dim1;
		i__4 = j + (i__ + m1) * a_dim1;
		t[i__3].r = a[i__4].r, t[i__3].i = a[i__4].i;
	    }
	}

	ktrmm_("R", "U", "C", "U", &m1, &m2, &c_b1, &a[i1 + i1 * a_dim1], lda,
		 &t[i1 * t_dim1 + 1], ldt);

	i__1 = *n - *m;
	kgemm_("N", "C", &m1, &m2, &i__1, &c_b1, &a[j1 * a_dim1 + 1], lda, &a[
		i1 + j1 * a_dim1], lda, &c_b1, &t[i1 * t_dim1 + 1], ldt);

	z__1.r = -1., z__1.i = -0.;
	ktrmm_("L", "U", "N", "N", &m1, &m2, &z__1, &t[t_offset], ldt, &t[i1 *
		 t_dim1 + 1], ldt)
		;

	ktrmm_("R", "U", "N", "N", &m1, &m2, &c_b1, &t[i1 + i1 * t_dim1], ldt,
		 &t[i1 * t_dim1 + 1], ldt);



/*        Y = (Y1,Y2); L = [ L1            0  ];  T = [T1 T3] */
/*                         [ A(1:N1,J1:N)  L2 ]       [ 0 T2] */

    }

    return;

/*     End of ZGELQT3 */

} /* kgelqt3_ */

