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
static halfcomplex c_b2 = {0.,0.};
static integer c__1 = 1;

/* > \brief \b ZGEQRT2 computes a QR factorization of a general doublereal or complex matrix using the compact WY re
presentation of Q. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGEQRT2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgeqrt2
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgeqrt2
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgeqrt2
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGEQRT2( M, N, A, LDA, T, LDT, INFO ) */

/*       INTEGER   INFO, LDA, LDT, M, N */
/*       COMPLEX*16   A( LDA, * ), T( LDT, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGEQRT2 computes a QR factorization of a complex M-by-N matrix A, */
/* > using the compact WY representation of Q. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= N. */
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
/* >          On entry, the complex M-by-N matrix A.  On exit, the elements on and */
/* >          above the diagonal contain the N-by-N upper triangular matrix R; the */
/* >          elements below the diagonal are the columns of V.  See below for */
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

/* > \date December 2016 */

/* > \ingroup complex16GEcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The matrix V stores the elementary reflectors H(i) in the i-th column */
/* >  below the diagonal. For example, if M=5 and N=3, the matrix V is */
/* > */
/* >               V = (  1       ) */
/* >                   ( v1  1    ) */
/* >                   ( v1 v2  1 ) */
/* >                   ( v1 v2 v3 ) */
/* >                   ( v1 v2 v3 ) */
/* > */
/* >  where the vi's represent the vectors which define H(i), which are returned */
/* >  in the matrix A.  The 1's along the diagonal of V are not stored in A.  The */
/* >  block reflector H is then given by */
/* > */
/* >               H = I - V * T * V**H */
/* > */
/* >  where V**H is the conjugate transpose of V. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  kgeqrt2_(integer *m, integer *n, halfcomplex *a, 
	integer *lda, halfcomplex *t, integer *ldt, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, t_dim1, t_offset, i__1, i__2, i__3;
    halfcomplex z__1, z__2;

    /* Local variables */
    integer i__, k;
    halfcomplex aii, alpha;
    extern void  kgerc_(integer *, integer *, halfcomplex *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *), kgemv_(char *, integer *, integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *, 
	    integer *, halfcomplex *, halfcomplex *, integer *), 
	    ktrmv_(char *, char *, char *, integer *, halfcomplex *, 
	    integer *, halfcomplex *, integer *), 
	    xerbla_(char *, integer *), klarfg_(integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments */

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
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -4;
    } else if (*ldt < f2cmax(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGEQRT2", &i__1);
	return;
    }

    k = f2cmin(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elem. refl. H(i) to annihilate A(i+1:m,i), tau(I) -> T(I,1) */

	i__2 = *m - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	klarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[f2cmin(i__3,*m) + i__ * a_dim1]
		, &c__1, &t[i__ + t_dim1]);
	if (i__ < *n) {

/*           Apply H(i) to A(I:M,I+1:N) from the left */

	    i__2 = i__ + i__ * a_dim1;
	    aii.r = a[i__2].r, aii.i = a[i__2].i;
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = 1., a[i__2].i = 0.;

/*           W(1:N-I) := A(I:M,I+1:N)^H * A(I:M,I) [W = T(:,N)] */

	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    kgemv_("C", &i__2, &i__3, &c_b1, &a[i__ + (i__ + 1) * a_dim1], 
		    lda, &a[i__ + i__ * a_dim1], &c__1, &c_b2, &t[*n * t_dim1 
		    + 1], &c__1);

/*           A(I:M,I+1:N) = A(I:m,I+1:N) + alpha*A(I:M,I)*W(1:N-1)^H */

	    d_cnjg(&z__2, &t[i__ + t_dim1]);
	    z__1.r = -z__2.r, z__1.i = -z__2.i;
	    alpha.r = z__1.r, alpha.i = z__1.i;
	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    kgerc_(&i__2, &i__3, &alpha, &a[i__ + i__ * a_dim1], &c__1, &t[*n 
		    * t_dim1 + 1], &c__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    a[i__2].r = aii.r, a[i__2].i = aii.i;
	}
    }

    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	i__2 = i__ + i__ * a_dim1;
	aii.r = a[i__2].r, aii.i = a[i__2].i;
	i__2 = i__ + i__ * a_dim1;
	a[i__2].r = 1., a[i__2].i = 0.;

/*        T(1:I-1,I) := alpha * A(I:M,1:I-1)**H * A(I:M,I) */

	i__2 = i__ + t_dim1;
	z__1.r = -t[i__2].r, z__1.i = -t[i__2].i;
	alpha.r = z__1.r, alpha.i = z__1.i;
	i__2 = *m - i__ + 1;
	i__3 = i__ - 1;
	kgemv_("C", &i__2, &i__3, &alpha, &a[i__ + a_dim1], lda, &a[i__ + i__ 
		* a_dim1], &c__1, &c_b2, &t[i__ * t_dim1 + 1], &c__1);
	i__2 = i__ + i__ * a_dim1;
	a[i__2].r = aii.r, a[i__2].i = aii.i;

/*        T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I) */

	i__2 = i__ - 1;
	ktrmv_("U", "N", "N", &i__2, &t[t_offset], ldt, &t[i__ * t_dim1 + 1], 
		&c__1);

/*           T(I,I) = tau(I) */

	i__2 = i__ + i__ * t_dim1;
	i__3 = i__ + t_dim1;
	t[i__2].r = t[i__3].r, t[i__2].i = t[i__3].i;
	i__2 = i__ + t_dim1;
	t[i__2].r = 0., t[i__2].i = 0.;
    }

/*     End of ZGEQRT2 */

    return;
} /* kgeqrt2_ */

