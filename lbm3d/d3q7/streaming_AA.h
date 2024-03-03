#pragma once

// A-A pattern
template < typename TRAITS >
struct D3Q7_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void postCollisionStreaming(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (SD.even_iter) {
			// write to the same lattice site, but the opposite DF direction
			SD.df(df_cur,mzz,x,y,z) = KS.f[pzz];
			SD.df(df_cur,zmz,x,y,z) = KS.f[zpz];
			SD.df(df_cur,zzm,x,y,z) = KS.f[zzp];
			SD.df(df_cur,zzz,x,y,z) = KS.f[zzz];
			SD.df(df_cur,zzp,x,y,z) = KS.f[zzm];
			SD.df(df_cur,zpz,x,y,z) = KS.f[zmz];
			SD.df(df_cur,pzz,x,y,z) = KS.f[mzz];
		}
		else {
			// write to the neighboring lattice sites, same DF direction
			SD.df(df_cur,pzz,xp, y, z) = KS.f[pzz];
			SD.df(df_cur,zpz, x,yp, z) = KS.f[zpz];
			SD.df(df_cur,zzp, x, y,zp) = KS.f[zzp];
			SD.df(df_cur,zzz, x, y, z) = KS.f[zzz];
			SD.df(df_cur,zzm, x, y,zm) = KS.f[zzm];
			SD.df(df_cur,zmz, x,ym, z) = KS.f[zmz];
			SD.df(df_cur,mzz,xm, y, z) = KS.f[mzz];
		}
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streaming(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (SD.even_iter) {
			// read from the same lattice site, same DF direction
			for (int i=0; i<7; i++)
				KS.f[i] = SD.df(df_cur,i,x,y,z);
		}
		else {
			// read from the neighboring lattice sites, but the opposite DF direction
			KS.f[mzz] = SD.df(df_cur,pzz,xp, y, z);
			KS.f[zmz] = SD.df(df_cur,zpz, x,yp, z);
			KS.f[zzm] = SD.df(df_cur,zzp, x, y,zp);
			KS.f[zzz] = SD.df(df_cur,zzz, x, y, z);
			KS.f[zzp] = SD.df(df_cur,zzm, x, y,zm);
			KS.f[zpz] = SD.df(df_cur,zmz, x,ym, z);
			KS.f[pzz] = SD.df(df_cur,mzz,xm, y, z);
		}
	}
};
