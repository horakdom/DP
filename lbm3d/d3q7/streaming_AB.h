#pragma once

// pull-scheme
template < typename TRAITS >
struct D3Q7_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void postCollisionStreaming(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// no streaming actually, write to the (x,y,z) site
		for (int i=0;i<7;i++) SD.df(df_out,i,x,y,z) = KS.f[i];
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streaming(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[mzz] = SD.df(df_cur,mzz,xp, y, z);
		KS.f[zmz] = SD.df(df_cur,zmz, x,yp, z);
		KS.f[zzm] = SD.df(df_cur,zzm, x, y,zp);
		KS.f[zzz] = SD.df(df_cur,zzz, x, y, z);
		KS.f[zzp] = SD.df(df_cur,zzp, x, y,zm);
		KS.f[zpz] = SD.df(df_cur,zpz, x,ym, z);
		KS.f[pzz] = SD.df(df_cur,pzz,xm, y, z);
	}
};
