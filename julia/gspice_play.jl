#using Revise
using Plots
using BenchmarkTools
using LinearAlgebra
using Statistics
using Printf

push!(LOAD_PATH, pwd()*"/julia")
using gspice



using FITSIO

# start ds9 instance
using SAOImageDS9, XPA
function ds9(image;min=0,max=100)
    if isnothing(XPA.find(r"^DS9:")) throw("Must start DS9!") end
    SAOImageDS9.draw(image',min=min,max=max)
    return
end

function ds9(image::BitMatrix)
    if isnothing(XPA.find(r"^DS9:")) throw("Must start DS9!") end
    SAOImageDS9.draw(Int8.(image'),min=0,max=1)
    return
end


function readit()
    path = "/Users/dfink/pcat/"
    fname = path*"frame-r-002583-2-0136.fits"
    ff = FITS(fname)
    im_in = read(ff[1])'
    img = im_in[480:879,405:804]
    return img
end

function get_psf(Nx=25, σ=1.2)
    Nx2 = (Nx+1)÷2
    g = zeros(Nx,Nx)
    for i=1:Nx
        for j=1:Nx
            r2 = (i-Nx2)^2+(j-Nx2)^2
            g[i,j] = exp(-r2/(2*σ*σ))
        end
    end
    return g
end


function star_image(;Np=512,Nstar=100)
    Nx  = 25
    Nx2 = (Nx+1)÷2
    g = get_psf(Nx)

    img = zeros(Np,Np)
    x = round.(Int,rand(Nstar).*(Np-Nx).+Nx2)
    y = round.(Int,rand(Nstar).*(Np-Nx).+Nx2)
    for k=1:Nstar
        img[x[k]-(Nx2-1):x[k]+(Nx2-1),y[k]-(Nx2-1):y[k]+(Nx2-1)] .+= g
    end
    return img
end


function getcorr(img)
    Nrow, Ncol = size(img)
    Np = 25
    Nx2 = (Np+1)÷2
    rows = collect(Np:Nrow-Np)
    cols = collect(Np:Ncol-Np)
    Nrow = length(rows)
    Ncol = length(cols)
    foo = zeros(Np*Np,Nrow*Ncol)
    k = 1
    for i in rows
        for j in cols
            sub = img[i-(Nx2-1):i+(Nx2-1),j-(Nx2-1):j+(Nx2-1)]
            foo[:,k] = reshape(sub,Np*Np)
            k+=1
        end
    end
    foo .-= mean(foo,dims=2)
    return foo
end

img = readit()
cimg = getcorr(img)
Npix, Nsam = size(cimg)
C = (cimg*cimg')./(Nsam-1)
sC = sqrt(C)
psfm = reshape(sC[313,:],Np,Np)
psfm ./= maximum(psfm)
img[201:225,176:200].+= psfm

# asymmetry
img = readit()
img .+= circshift(img,(-2,0)).*0.5
img .+= circshift(img,(-4,0)).*0.25

imgs = wt .* img
cimg = getcorr(imgs)
Npix, Nsam = size(cimg)
C = (cimg*cimg')./(Nsam-1)
sC = sqrt(C)
psfm = reshape(sC[313,:],Np,Np)
psfm ./= maximum(psfm)

# 2021-Dec-18  This all kinda works, but returns a symmetric PSF.
# Not sure how to handle an asymmetric one.




function Sersic(A, x0, n, Cinv; xlim=(-4,4), ylim=(-4,4))
	# see https://academic.oup.com/mnras/article/441/3/2528/1108396
	Nx  = 32
	Ny  = 32
	Δx  = (xlim[2]-xlim[1])/Nx
	Δy  = (ylim[2]-ylim[1])/Ny
	img = zeros(Nx+1,Ny+1)
	xvals = collect(xlim[1]:Δx:xlim[2])
	yvals = collect(ylim[1]:Δy:ylim[2])
	η = 0.5/n
	k = 1.9992*n - 0.3271
	#Σinv = inv(Σ)
	for j = 1:Nx+1
		for i = 1:Ny+1
			R = [xvals[j],yvals[i]] - x0
			img[i,j] = exp(-k*(R'*Cinv*R)^η)
		end
	end
	img .*= (A/sum(img))
	return img
end


function generate1random()
	flux = rand()*10 + 1
	cen = randn(2).*0.125
	n   = rand() + 1   # 1 to 2
	θ   = rand()*π
	rot = reshape([cos(θ), sin(θ), -sin(θ), cos(θ)],2,2)
	cov = 16 .*(rot' * diagm([1.0,rand()*4+0.5]) * rot)
	img = Sersic(flux, cen, n, inv(cov))
	return img
end


function interloper(cen)
	flux = 3.0
	#cen = randn(2).*0.125
	n   = 1.5   # 1 to 2
	θ   = rand()*π
	rot = reshape([cos(θ), sin(θ), -sin(θ), cos(θ)],2,2)
	cov = 16 .*(rot' * diagm([1.0,rand()*4+0.5]) * rot)
	img = Sersic(flux, cen, n, inv(cov))
	return img
end




function pred_gal(covmat, k, kstar, Dvec)
	cov_kk         = covmat[k, k]
# RENAME ???
	cov_kkstar     = covmat[kstar, k] # dim [nkstar,nk]
	#cov_kstark     = covmat[k, kstar]
	#cov_kstarkstar = covmat[kstar, kstar]

# -------- Choleksy inversion
	icov_kk = inv(cholesky(cov_kk))

# -------- compute the prediction covariance (See RW, Chap. 2)
	#predcovar = cov_kstarkstar - (cov_kkstar*(icov_kk*cov_kstark))
	predkstar = cov_kkstar * (icov_kk * Dvec[k])
	return predkstar
end



function halfplane_mask(θ; Nx = 32, Ny = 32, xlim=(-4,4), ylim=(-4,4))
	# θ = 0 -> right half is masked
	# θ = π/2 -> top half is masked
	Δx  = (xlim[2]-xlim[1])/Nx
	Δy  = (ylim[2]-ylim[1])/Ny
	msk = Array{Bool,2}(undef, (Nx+1,Ny+1))
	xvals = collect(xlim[1]:Δx:xlim[2])
	yvals = collect(ylim[1]:Δy:ylim[2])
	uvec = [cos(θ),sin(θ)]
	for j = 1:Nx+1
		xu = xvals[j]*uvec[1]
		for i = 1:Ny+1
			msk[i,j] = (xu+yvals[i]*uvec[2]) > 1E-10
		end
	end
	return msk
end



Ngal = 50000
Np = 33
cub = zeros(Np, Np, Ngal)
for i=1:Ngal cub[:,:,i] = generate1random() end
ns = randn((Np*Np, Ngal)).*0.001
dat = ns+reshape(cub, (Np*Np, Ngal))
cov = Symmetric(dat*dat')./Ngal
#cov = Symmetric(cov)

# interloper cov
cub = zeros(Np, Np, Ngal)
for i=1:Ngal cub[:,:,i] = interloper([2,0]+randn(2).*0.125) end
ns = randn((Np*Np, Ngal)).*0.001
dat2 = ns+reshape(cub, (Np*Np, Ngal))
cov2 = Symmetric(dat2*dat2')./Ngal
#cov2 = Symmetric(cov2)


k = collect(1:545)
kstar = collect(546:1089)
Dvec = dat[:,5]


function do4(dat,cov,ind,k,kstar)
	Dvec = dat[:, ind]
	predpix = pred_gal(cov, k, kstar, Dvec)
	g1 = reshape(Dvec,Np,Np)
	g2 = copy(g1)
	g2[kstar] = predpix
	g3 = copy(g1)
	g3[kstar] .= 0.0
	all = [[g2; g1] [g2-g1; g3]]
	return all
end


function do6(dat,cov,ind,k,kstar)
	Dvec = copy(dat[:, ind])
	predpix = pred_gal(cov, k, kstar, Dvec)
	g1 = reshape(copy(Dvec),Np,Np)
	g2 = copy(g1)
	g2[kstar] = predpix
	g3 = copy(g1)
	g3[kstar] .= 0.0
	# corrupt the data a little bit
	Dvec[Np*13+10:Np*13+12] .+= 0.02
	predpix = pred_gal(cov, k, kstar, Dvec)
	g5 = reshape(Dvec,Np,Np)
	g6 = copy(g5)
	g6[kstar] = predpix
	all = [[g2; g1] [g2-g1; g3] [g5-g6; g5]]
	return all
end


function doit(dat,cov,ind,k,kstar)
	Dvec = copy(dat[:, ind])
	predpix = pred_gal(cov, k, kstar, Dvec)
	g1 = reshape(copy(Dvec),Np,Np)
	g2 = copy(g1)
	g2[kstar] = predpix
	g3 = copy(g1)
	g3[kstar] .= 0.0
	# corrupt the data a little bit
	gal2 = interloper([2,0])
	g4 = gal2
	Dvec += reshape(gal2, (Np*Np))
	predpix = pred_gal(cov, k, kstar, Dvec)
	g5 = reshape(Dvec,Np,Np)
	g6 = copy(g5)
	g6[kstar] = predpix
	all = [[g2; g1] [g2-g1; g5] [g5-g6; g4]]
	return all
end

ind=1
heatmap(doit(dat,cov,ind+=1,k,kstar),aspect_ratio=1)


function deblend_central(dat,cov,ind,ang)
	Dvec  = copy(dat[:, ind])
	minDvec = copy(Dvec)
	for i=1:8
		ang = i*π/4
		msk = reshape(halfplane_mask(ang),Np*Np)
		k★  = findall(msk)
		k   = findall(.~msk)
		predpix = pred_gal(cov, k, k★, Dvec)
		minDvec[k★] = min.(minDvec[k★], predpix)
	end
	g1 = reshape(copy(Dvec),Np,Np)
	g2 = reshape(copy(minDvec),Np,Np)

	g3 = copy(g1)
	#g3[k★] .= 0.0
	# corrupt the data a little bit
	gal2 = interloper(rand(2).*6 .-3)
	g4 = gal2
	Dvec += reshape(gal2, (Np*Np))

	#shift a bit just to really challenge it
	minDvec = copy(circshift(Dvec,0))
	for i=1:8
		ang = i*π/4
		msk = reshape(halfplane_mask(ang),Np*Np)
		k★  = findall(msk)
		k   = findall(.~msk)
		predpix = pred_gal(cov, k, k★, Dvec)
		minDvec[k★] = min.(minDvec[k★], predpix)
	end

	#predpix = pred_gal(cov, k, kstar, Dvec)
	g5 = reshape(Dvec,Np,Np)
	g6 = reshape(copy(minDvec),Np,Np)
	#g6[kstar] = predpix
	all = [[g6; g1] [g2-g1; g5] [g5-g6; g4]]
	return all
end



function gauss_pdf_prod(μ₁,Σ₁,μ₂,Σ₂)
    Σ₁₂inv = inv(Σ₁+Σ₂)
    μ =  Σ₁₂inv * (Σ₂*μ₁ + Σ₁*μ₂)
    Σ = Σ₁*Σ₂ * Σ₁₂inv
    return μ,Σ
end


# 124 ms on Mac, 233 ms for 1000
function gauss_deblend(x_d, Σ_a, Σ_b)
	Σ_ab_inv = inv(cholesky(Σ_a + Σ_b))
	Σ_ad = Σ_a*Σ_ab_inv*Σ_b  # 80 ms

	Σinvx = (Σ_ab_inv*x_d)
	μ_ad = Σ_a*Σinvx
	μ_bd = Σ_b*Σinvx
	return μ_ad, μ_bd, Σ_ad
end

# 140 ms 290 ms for 1000
function gauss_deblend(x_d, Σ_a, Σ_b, μ_a, μ_b)
	Σ_ab_inv = inv(cholesky(Σ_a + Σ_b))
	Σ_ad = Σ_a*Σ_ab_inv*Σ_b

	μ_a0 = (Σ_b - Σ_a) * (Σ_ab_inv * μ_a)
	μ_b0 = (Σ_a - Σ_b) * (Σ_ab_inv * μ_b)

	μ_ad = μ_a0 .+ Σ_a*(Σ_ab_inv*x_d)
	μ_bd = μ_b0 .+ Σ_b*(Σ_ab_inv*x_d)
	return μ_ad, μ_bd, Σ_ad
end


function junk()
	μ_a = zeros(Np*Np)
	μ_b = zeros(Np*Np)
	Σ_a = copy(cov)
	Σ_b = copy(cov2)

	ind = 5
	cen = [2.,0.5]
	Dvec  = copy(dat[:, ind])
	x_d = Dvec + reshape(interloper(cen),33*33)
	μ_ad, μ_bd, Σ_ad = gauss_deblend(x_d, cov, cov2, μ_a, μ_a)

	heatmap(reshape(x_d,(33,33)),aspect_ratio=1)
	heatmap(reshape(μ_ad,(33,33)),aspect_ratio=1)
	heatmap(reshape(μ_bd,(33,33)),aspect_ratio=1)
	heatmap(reshape(sqrt.(diag(Σ_ad)),(33,33)),aspect_ratio=1)
end


function covshift(cov, Δrow, Δcol)
	cov2  = zeros(size(cov))
	covshift!(cov2, cov, Δrow, Δcol)
	return cov2
end


# shift covariance matrix to a different center
function covshift!(cov2, cov, Δrow, Δcol)
	diagval = 1e-6
	row1  = clamp(1-Δrow,1,Np)
	rowNp = clamp(Np-Δrow,1,Np)
	col1  = clamp(1-Δcol,1,Np)
	colNp = clamp(Np-Δcol,1,Np)

	for jcol=col1:colNp
		for jrow=row1:rowNp
			j1=jrow+Np*(jcol-1)
			j2=jrow+Δrow + Np*(jcol+Δcol-1)
			c1=(j1-1)*Np*Np-Np
			c2=Δrow + Np*(Δcol-1) + (j2-1)*Np*Np
			for icol=col1:colNp
				d1 = Np*icol + c1
				d2 = Np*icol + c2
				# for irow=clamp(1-Δrow,1,Np):clamp(Np-Δrow,1,Np)
				# 	i1=irow+d1
				# 	i2=irow+d2
				# 	cov2[i2] = cov[i1]
				# end
				cov2[row1+d2:rowNp+d2] = cov[row1+d1:rowNp+d1]
			end
		end
	end
	for k=1:Np*Np
		if (cov2[k,k] == 0) cov2[k,k] = diagval end
	end
	return cov2
end


function gdb_TS(x_d, cov, cens)
	scov = zeros(size(cov))
	for k=1:length(cens)
		cen = cens[k]
		scov .+= covshift(cov,round(Int,cen[2]*4), round(Int,cen[1]*4))
	end
	TS = x_d' * (scov\x_d)
	return TS
end




function gddemo3(cov, Ngal=3)

	# list of galaxy centers
	cens = [[0.0,0.0]]
	for k=2:Ngal
		cenk = rand(2).*6 .-3
		append!(cens,[cenk])
	end

	# summed shifted covariance
	scov = zeros(size(cov))
	for k=1:Ngal
		cen = cens[k]
		scov .+= covshift(cov,round(Int,cen[2]*4), round(Int,cen[1]*4))
	end

	# galaxy images
	gal = zeros(Np,Np,Ngal)
	sumgal = zeros(Np,Np)
	for k=1:Ngal
		gal[:,:,k] = interloper(cens[k])
		sumgal += gal[:,:,k]
	end

	# data vector
	x_d  = copy(reshape(sumgal, 33*33))  # observed blend
	x_d .+= randn(33*33)*0.001

	CinvXd = scov\x_d

	μ_gal = zeros(Np,Np,Ngal)
	for k=1:Ngal
		cen = cens[k]
		cov_k = covshift(cov,round(Int,cen[2]*4), round(Int,cen[1]*4))
		μ_gal[:,:,k] = cov_k * CinvXd
	end

	sum_μ = sum(μ_gal,dims=3)[:,:,1]
	grid = [sum_μ-sumgal; sum_μ; sumgal]
	for k=1:Ngal
		gk = [μ_gal[:,:,k]-gal[:,:,k]; μ_gal[:,:,k]; gal[:,:,k]]
		grid = [grid gk]
	end
	return grid
end



function gddemo4(cov, Ngal=3)

	# list of galaxy centers
	cens = [[0.0,0.0]]
	for k=2:Ngal
		cenk = rand(2).*6 .-3
		append!(cens,[cenk])
	end

	# summed shifted covariance
	scov = zeros(size(cov))
	for k=1:(Ngal-1)
		cen = cens[k]
		scov .+= covshift(cov,round(Int,cen[2]*4), round(Int,cen[1]*4))
	end
	noisecov = 4E-6 * I
	scov += noisecov

	# galaxy images
	gal = zeros(Np,Np,Ngal)
	sumgal = zeros(Np,Np)
	for k=1:(Ngal-1)
		gal[:,:,k] = interloper(cens[k])
		sumgal += gal[:,:,k]
	end
	noisegal = randn(Np*Np)*0.001
	sumgal += reshape(noisegal,Np,Np)
	gal[:,:,Ngal] = noisegal
	# data vector
	x_d  = copy(reshape(sumgal, 33*33))  # observed blend
	#x_d += noisegal

	CinvXd = scov\x_d

	μ_gal = zeros(Np,Np,Ngal)
	for k=1:(Ngal-1)
		cen = cens[k]
		cov_k = covshift(cov,round(Int,cen[2]*4), round(Int,cen[1]*4))
		μ_gal[:,:,k] = cov_k * CinvXd
	end
	μ_gal[:,:,Ngal] = noisecov * CinvXd

	sum_μ = sum(μ_gal,dims=3)[:,:,1]
	grid = [sum_μ-sumgal; sum_μ; sumgal]
	for k=1:Ngal
		gk = [μ_gal[:,:,k]-gal[:,:,k]; μ_gal[:,:,k]; gal[:,:,k]]
		grid = [grid gk]
	end
	return grid
end




function gddemo(dat,cov,cov2,ind)
	cen = [2.0,0.0]
	#cen = rand(2).*4 .-2
	#shcov = covshift(cov, round(Int,cen[2]*4), round(Int,cen[1]*4))
	shcov=cov2
	gal1 = copy(dat[:, ind])
	gal2 = interloper(cen)
	x_d  = gal1 + reshape(gal2, 33*33)  # observed blend
	μ_ad, μ_bd, Σ_ad = gauss_deblend(x_d, cov, shcov, μ_a, μ_a)

	g1 = reshape(copy(gal1),Np,Np)
	g2 = reshape(μ_ad,Np,Np)

	g5 = reshape(copy(gal2),Np,Np)
	g6 = reshape(μ_bd,Np,Np)

	g3 = g1+g5
	g4 = (g1+g5)-(g2+g6)

	all = [[g2; g1] [g4; g3] [g6; g5]]
	return all
end


heatmap(gddemo(dat,cov,cov2,ind+=1),aspect_ratio=1)


function gimage(μ, Σ; xlim=(-10,10), ylim=(-10,10))
	Nx  = 200
	Ny  = 200
	Δx  = (xlim[2]-xlim[1])/Nx
	Δy  = (ylim[2]-ylim[1])/Ny
	img = zeros(Nx+1,Ny+1)
	xvals = collect(xlim[1]:Δx:xlim[2])
	yvals = collect(ylim[1]:Δy:ylim[2])
	Σinv = inv(Σ)
	for j = 1:Nx+1
		for i = 1:Ny+1
			R = [xvals[j],yvals[i]] - μ
			img[i,j] = exp(-0.5*(R'*Σinv*R))
		end
	end
	return img
end



Σ₁ =reshape([9.0, 1., 1., 1.0],2,2)
μ₁ = [0,2]
Σ₂ = RR'*Σ₁*RR
μ₂ = [6,0]

μ,Σ = gauss_pdf_prod(μ₁,Σ₁,μ₂,Σ₂)

heatmap(gimage(μ₁, Σ₁)+gimage(μ₂, Σ₂),aspect_ratio=1)
heatmap(gimage(μ, Σ),aspect_ratio=1)

heatmap(Sersic(10.0, [0,0], 1.0, inv(Σ₁)),aspect_ratio=1)

using Dates
using BenchmarkTools
using LinearAlgebra
using Statistics

function image_model_eval!(image::Matrix{Float32}, x::Array{Float32},
			               y::Array{Float32}, flux::Array{Float32},
						   back::Float32, cf::Matrix{Float32})

	recon = nothing
    image_model_eval!(image, x, y, flux, back, cf, recon)
	return
end

function image_model_eval!(image::Matrix{Float32}, x::Array{Float32},
			               y::Array{Float32}, flux::Array{Float32},
						   back::Float32, cf::Matrix{Float32}, recon=nothing)

	# -------- timing
	t0 = now()

	if isnothing(image) throw("Must pre-allocate image array") end

	imsz = size(image)
	NX   = Int32(imsz[2])
	NY   = Int32(imsz[1])

	# -------- fill with zeros
	#fill!(image, 0f0)

	# -------- should pass these somehow
	nstar = Int32(length(x))
	nc    = Int32(25)
	rad   = Int32(nc÷2)
	if isnothing(recon) recon = Array{Float32,2}(undef, nc*nc, nstar) end

	# -------- integer and fractional parts of (x,y)
	ix = floor.(Int32,x)
	dx = 1.0f0 .- (x-ix)   #  fix this!!!
	iy = floor.(Int32,y)
	dy = 1.0f0 .- (y-iy)

	# -------- array of offset powers
	# this takes ~ 1/10 of the total time, could probably be sped up
	dd = [ones(Float32,length(dx)) dx dy dx.*dx dx.*dy dy.*dy dx.*dx.*dx dx.*dx.*dy dx.*dy.*dy dy.*dy.*dy]
	dd .*= flux

	# the malloc for recon takes 1/10th of the time...
	#recon = (cf' * dd')  #  (NX*NY x Nstar)
	mul!(recon,cf',dd')

	for istar = 1:nstar
		xx = ix[istar]  # integer coords, 1 indexed
		yy = iy[istar]
		imax = min(xx+rad, NX)
		jmax = min(yy+rad, NY)
		for j = max(Int32(1),yy-rad):jmax
			j2 = (Int32(istar-1)*nc+j-yy+rad)*nc
			for i = max(Int32(1),xx-rad):imax
				i2 = i-xx+rad+Int32(1)
				@inbounds image[j,i] += recon[i2+j2]
			end
		end
	end

	dt = (now()-t0).value
	# println("T1: ", dt, " ms")

	# -------- add background
	if back !=0
		image .+= back
	end

  	return
end


using Plots
function testit(cff,nstar=1000)

	T = Float32
	x = rand(T, nstar) .*70 .+15
	y = rand(T, nstar) .*70 .+15

  	flux = rand(T, nstar)

  	nc   = 25
  	imsz = (100, 100)

  	nstar = Int32.(round.(rand(100) .* 950 .+10))
    t = zeros(100)
    for i=1:100
		x = rand(T, nstar[i]) .*70 .+15
		y = rand(T, nstar[i]) .*70 .+15
	  	flux = rand(T, nstar[i])

		image1 = zeros(T, imsz)
		recon  = zeros(Float32, (nc*nc, nstar[i]))
		t[i] = @elapsed image_model_eval!(image1, x, y, flux, back, cf, recon)
	end

  	p = scatter(nstar, t*1000, xlabel="N star",
	            ylabel="time [ms]") #, ps=3, ytit='time [ms]'
	plot!([0,500],[0,0.8])
	display(p)

  	return
end



function pcat_model_likelihood(x, y, flux, back, cff, mock)

	imsz  = size(mock)
	recon = nothing
	model = zeros(Float32, imsz)
	image_model_eval!(model, x, y, flux, back, cff, recon)

  	chi2  = sum((model-mock).^2)/(1.0^2)
	lnL  = -0.5*chi2

  	return lnL
end



T = Float32
nstar = 1000
x = rand(T, nstar) .*70 .+15
y = rand(T, nstar) .*70 .+15
flux = rand(T, nstar)

# -------- model image
nc   = 25
npar = 10
#cff = float(reform(cf, nc*nc, npar))
cf = ones(T,npar,nc*nc)

imsz = (100, 100)
back = 0f0

image1 = zeros(T, imsz)
recon = zeros(Float32, nc*nc, nstar)
image_model_eval!(image1, x, y, flux, back, cf, recon)

using FITSIO
f = FITS("/Users/dfink/gspice/julia/psf-pad-0.fits")
idl_model = read(f[1])'
idl_x = read(f[2])
idl_y = read(f[3])
idl_flux = read(f[4])
idl_cf = read(f[5])
idl_cff = read(f[6])'
cff = copy(idl_cff)
imsz = (100,150)
back = 0f0
image3 = zeros(Float32, imsz)
image_model_eval!(image3, idl_x.+1, idl_y.+1, idl_flux, back, copy(idl_cff), recon)


println(1)










function random_pow(T, N, alpha, x0, x1)

	if alpha == -1 throw("Do not call with alpha=-1") end
	alpha1 = convert(T, alpha+one(T))
	y = rand(T, N)
	x = ((x1^alpha1-x0^alpha1).*y .+ (x0^alpha1)).^(one(T)/alpha1)

	return x
end



function where(cond)
    ind = findall(cond)
    num = length(ind)
    return ind, num
end


function sample_from_distribution(x)

  xcumul = cumsum(reshape(x,length(x)))
  xcmax  = xcumul[end]
  cut    = rand()*xcmax
  w,nw   = where(xcumul .> cut)
  return w[1]
end

function myfindmax(x)
	T    = typeof(x[1])
	ind  = zero(T)
	xmax = convert(T,-1e30)
	for i=1:length(x)
		if x[i] > xmax
			xmax = x[i]
			ind = i
		end
	end
	return ind
end

function myfindmaxdiff(x,y)
	T    = typeof(x[1])
	ind  = zero(T)
	xmax = convert(T,-1e30)
	for i=1:length(x)
		if (x[i]-y[i]) > xmax
			xmax = x[i]-y[i]
			ind = i
		end
	end
	return ind
end


println(1)

# This does the MCMC
#function mc(data, model, xcat, ycat, cen, str, cff)
function mc(cff, data=nothing)

	tstart = now()
	unwise = false
	tile = "1497p015"
	T = Float32
	image_ivar = unwise ? 1.0/1.6^2 : 1.0

	# -------- read the 125x125 upsampled PSF
	if unwise getpsf(psf) end

# -------- make a model to fit
	if isnothing(data)
		imsz  = (320, 200)         # nrow, ncol
		nstar = 4000

		foo = Dict()
		foo["x"] = rand(T,nstar) .* imsz[2]
		foo["y"] = rand(T,nstar) .* imsz[1]
		foo["flux"] = random_pow(T, nstar, -2.0, 3, 1000)
		foo["back"] = 0.0f0
		back = foo["back"]

		# -------- make mock image
		noise = randn(T, imsz[1], imsz[2]) .* 1f0
		mock0 = zeros(T, imsz)
		image_model_eval!(mock0,foo["x"], foo["y"], foo["flux"], foo["back"], cff)
		mock  = noise + mock0
	else
		mock = data
	end
	#println("WRONG posf")
	#mock = mock[:,end:-1:1]


	imsz = size(mock)
	weight = ones(T, imsz)
	resid = ones(T, imsz)

	# -------- read real image
	#  pcat_sdss_image, mock


	# -------- number of stars to use in fit.  Start with terrible guesses.
	nstar = 2000

	x = rand(T, nstar) .* 20
	y = rand(T, nstar) .* 20
	flux = zeros(T, nstar) .+ 0.1f0
	back = 0.0f0

	# -------- initial ln(L)
	lnL = pcat_model_likelihood(x, y, flux, back, cff, mock)
	plnL = copy(lnL)

	# -------- make output data structure
	nsam  = 500
	nloop = 400
	zz = zeros(T, nstar)

#	str = DataFrame(x=zz, y=zz, flux=zz, back=zz)

#	str = replicate({x:zz, y:zz, flux:zz, back:zz, acceptance: 0., nloop:0L, dt:0., dt1:0., dt2:0., dt3:0.}, nsam)

	crad = 10
	pad = 5

	for j=1:nsam
		t0 = now()
		dt1 = 0.
		dt2 = 0.
		dt3 = 0.
		dt4 = 0.
		acc  = falses(nloop)
		nmov = zeros(Int32, nloop)

		# -------- update model
		model = zeros(T, imsz)
		image_model_eval!(model, x, y, flux, back, cff)

		pmodel = copy(model)
		resid  = mock-model

		for i=1:nloop

		# -------- select step type
			rtype = rand()
			type = "hopper"
			if rtype > 0.3 type = "mover" end
			if rtype < 0.1 type = "trim" end



        	penalty = 1

        	if type=="trim"
           		nmov[i] = 2
           		nw = 2
				mover = floor(Int32, rand()*nstar)+1  # pick a star to combine with neighbor
				neighbordist = sqrt.((x .- x[mover]).^2+(y .- y[mover]).^2)
				wclose, nclose = where((neighbordist .> 1e-7) .& (neighbordist .< 1.9))
           		if nclose >= 1
					#               print, 'nclose', nclose, neighbordist[wclose]
					mover = [mover; wclose[1]]
					#              print, flux[mover]
					fx1 = flux[mover[1]]
					fx2 = flux[mover[2]]
					pflux = [fx1+fx2; 0.1f0]
					px = [(x[mover[1]]*fx1 + x[mover[2]]*fx2)/(fx1+fx2), rand(T)*imsz[2]]
					py = [(y[mover[1]]*fx1 + y[mover[2]]*fx2)/(fx1+fx2), rand(T)*imsz[1]]
					pback = copy(back)
					penalty = 1f-20   # always do it

				else
					type="mover"
        		end
			end


			if type=="merge"
           		nmov[i] = 2
           		nw = 2
				yx = myfindmaxdiff(mock,model)-1
				y_peak = clamp(mod(yx,imsz[1]), 1, imsz[1])+1
				x_peak = clamp(   (yx÷imsz[1]), 1, imsz[2])+1


				#mover = floor(Int32, rand()*nstar)+1  # pick a star to combine with neighbor
				neighbordist = sqrt.((x .- x_peak).^2+(y .- y_peak).^2)
				wclose, nclose = where((neighbordist .> 1e-7) .& (neighbordist .< 1.5))
           		if nclose >= 2
					println("        Nclose ",nclose)
					println((mock-model)[y_peak-2:y_peak+2,x_peak])
					#               print, 'nclose', nclose, neighbordist[wclose]
					mover = wclose[1:2]
					#              print, flux[mover]
					fx1 = flux[mover[1]]
					fx2 = flux[mover[2]]
					pflux = [fx1+fx2; 0.1f0]
					px = [(x[mover[1]]*fx1 + x[mover[2]]*fx2)/(fx1+fx2), rand(T)*imsz[2]]
					py = [(y[mover[1]]*fx1 + y[mover[2]]*fx2)/(fx1+fx2), rand(T)*imsz[1]]
					pback = copy(back)
					penalty = 1f-25   # always do it

				else
					type="mover"
        		end
			end



			if type=="mover"
				# -------- select stars to move
				cx = rand(T)*(imsz[2]-(crad-pad)*2)+(crad-pad)
				cy = rand(T)*(imsz[1]-(crad-pad)*2)+(crad-pad)
				mover, nw = where((abs.(cx.-x) .< crad) .& (abs.(cy.-y) .< crad))   # this is slow and stupid
				nmov[i] = nw

				# -------- propose new parameters
				if nw > 0
					dlnflux = randn(T, nw) .* 0.02f0
					pflux   = flux[mover] .* exp.(dlnflux)
					dpos_rms = 7.0f0 ./max.(flux[mover], pflux)./sqrt(40f0)
					#dpos_rms = 0.02f0
					dx = randn(T, nw).*dpos_rms
					dy = randn(T, nw).*dpos_rms

					 dback = 0.0f0
					#dback = randn(T)*0.0001f0

					# -------- proposed params
					px = clamp.(x[mover]+dx, 1-pad, imsz[2]+pad)
					py = clamp.(y[mover]+dy, 1-pad, imsz[1]+pad)
					pback = back+dback
				end
			end

			# dt1 += (systime(1)-t1)
			#
			# t4 = systime(1)
			if type=="hopper"
				resid = mock-model
				nmov[i] = 1
				nw = 1
				mover = floor(Int32, rand()*nstar)+one(Int32)  # pick a star to hop
				#           thisp = exp((resid >0)^2/100) * (resid GT 0)
				#           thisp = resid*resid * (resid GT 0)
				thisp = max.(resid, 0)
				# this might be messed up
				yx = sample_from_distribution(thisp)-1 # 0-indexed
				py = clamp(mod(yx,imsz[1])+rand(T), -pad, imsz[1]+pad)+1
				px = clamp(   (yx÷imsz[1])+rand(T), -pad, imsz[2]+pad)+1
				pflux = resid[yx+1]/0.143f0 # depends on psf!
				# what is this penalty??
				#penalty = thisp[px,py]/total(thisp)/sqrt(2)*imsz[0]*imsz[1]
				penalty = resid[yx+1]/sum(thisp)/sqrt(2)*imsz[1]*imsz[2]
				if pflux < 0 throw("negative pflux") end
				pback = copy(back)
				#           print, px, py, pflux
			end
# dt4 += (systime(1)-t4)
#
	        if type=="randomhopper"
				mover, nw = where(rand(T, nstar) .< 5.0f0/nstar)
				#println("mover count", nw)
				nmov[i] = nw
				if nw > 0
					px = rand(T, nw) .*imsz[2] .+1
					py = rand(T, nw) .*imsz[1] .+1
					pflux = flux[mover]
					pback = back
				end
				penalty = 1.0f0
	        end
#
			if nw > 0
#t2 = systime(1)
# -------- perturb model

				pmodel = copy(model)
				if length(size(px))==0
					image_model_eval!(pmodel, [px], [py], [pflux], pback-back, cff)
					image_model_eval!(pmodel, [x[mover]], [y[mover]], [-flux[mover]], 0f0, cff)
				else
					# image_model_eval!(pmodel, px, py, pflux, pback-back, cff)
					# image_model_eval!(pmodel, x[mover], y[mover], -flux[mover], 0f0, cff)
			 		image_model_eval!(pmodel, [px; x[mover]], [py; y[mover]], [pflux; -flux[mover]], 0f0, cff)
				end

				diff2 = 0.0
				for i=1:imsz[1]*imsz[2] diff2 += (pmodel[i]-mock[i])^2 end
				#diff2 = sum((pmodel-mock).^2)
	           	plnL  = -0.5*diff2*image_ivar
	           	accept = exp(clamp(plnL-lnL, -50, 50))/ penalty
	#dt2 += (systime(1)-t2)
	           	if accept > rand(T)
					#if type == "trim" println("move type ",type) end
					# println("mover  ",mover)
					#println("px  ",px)
				 	x[mover] = px
					y[mover] = py
					flux[mover] = pflux
					back = copy(pback)
					model = copy(pmodel)
					#resid = mock-model
					lnL = plnL
					acc[i] = true
				end
			end
        end  # over nloop
		if (j==1) | (mod(j,50) == 0)
			p = heatmap([model (mock-model)],clims=(-10,10))
			display(p)
		end

	#t3 = systime(1)

	#         if keyword_set(atv) AND (i eq 50) then begin
	#
	#            atv, [mock, mock-model+back], /al, /s
	# if keyword_set(xcat) then atvplot, xcat, ycat, ps=1, syms=0.7, color='green'
	#
	# #           atvplot, x-0.5, y-0.5, ps=7, syms=0.5
	#         endif
	# dt3 += (systime(1)-t3)
	#

     # str[j].x = x
     # str[j].y = y
     # str[j].flux = flux
     # str[j].back = back
     # str[j].acceptance = mean(float(acc))
     # str[j].nloop = nloop
     # str[j].dt = systime(1)-t0
     # str[j].dt1=dt1
     # str[j].dt2=dt2
     # str[j].dt3=dt3
	#println("Loop ", j, "  Acceptance fraction", mean(acc), "  lnL: ", lnL, "   plnL: ", plnL)
	@printf("Loop:%4d   Acc frac:%6.3f  lnL:%11.2f   plnL:%11.2f   back:%8.3f\n", j, mean(acc), lnL, plnL, back)
	dback = mean(mock-model)
	back += 0.1f0 * dback #- 0.01f0
#      print, 'Loop',  j, '  Acceptance fraction', str[j].acceptance,  '  back', back, '  dt', str[j].dt
#      print, 'DT4:  ', dt1, dt2, dt3, dt4
# #     print, 'Min flux: ', min(flux)
	end
  # fitsname = string('unwiser_'+tile+'_x', cen[0], 'y', cen[1], '.fits', format='(A,I4.4,A,I4.4,A)')
  #
  # print, 'Writing: ', fitsname
  # mwrfits, str, fitsname, /create
  # if ~unwise then mwrfits, foo, fitsname
  #
  # print, nloop, ' samples per loop'
  # print, 'T1 (proposal)       :', mean(str.dt1)
  # print, 'T2 (model and log L):', mean(str.dt2)
  # print, 'T3 (display)        :', mean(str.dt3)
  # print, 'All                 :', mean(str.dt)
    dt = (now()-tstart).value / 1000
  	println("Time: ", dt, " s")

  	return resid,x,y,flux,back
end

#time 200x320 1000stars 87s
#     100x160  500stars 39s, 28s, 28s, 23s, 21 with less heatmap


using Profile
Profile.clear()
@profile mc(cff)
Juno.profiler()

# test to see if psf_poly_fit is doing the correct thing

function psf_recon_test(psf0, cf, recon)

	sz = size(psf0, /dimen)
	npix = sz[0]
	if sz[1] NE npix then stop assert PSF is square

	# -------- pad out by 1 row and column
	#  psf = dblarr(npix+1, npix+1)
	#  psf[0:npix-1, 0:npix-1]=f

	# -------- design matrix A for each 5x5 sub region
	nbin = 5
	nc = npix/nbin


	# -------- fit the polynomial coefficients
	cf = psf_poly_fit(psf0, nbin=nbin)
	npar = (size(cf, /dimen))[2]

	# -------- now reconstruct to check
	recon = fltarr(npix, npix)

	cft = float(transpose(reform(cf, nc*nc, npar)))
	t1 = systime(1)
  for ix=0, nbin-1 do begin
     for iy=0, nbin-1 do begin
        dx = ix*0.2
        dy = iy*0.2
        dd = [1, dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy]

        recon[ix:*:nbin, iy:*:nbin] = reform(cft ## dd, nc, nc)
     endfor
  endfor
  print, 'Time: ', systime(1)-t1

  ddall = dd##(fltarr(100)+1)


  t1 = systime(1)
  bar = cft##ddall
  print, 'Time: ', systime(1)-t1
  t1 = systime(1)
  bar = cft##ddall
  print, 'Time: ', systime(1)-t1
  t1 = systime(1)
  bar = cft##ddall
  dt = systime(1)-t1
  print, 'Time: ', dt

  m = (size(cft, /dim))[1]
  k = (size(cft, /dim))[0]
  n = (size(ddall, /dim))[0]
	help, m, k, n
	#  a=findgen(k, m)
	#  b=findgen(n, k)

	niter=100
	t1 = systime(1)
	for i=0L, niter-1 do c=cft##ddall
	dt = systime(1)-t1
	print, (2LL*m*n*k*niter)/dt*1d-9, ' GFlops', '  dt = ', dt

	soname = filepath('libmatrix.'+idlutils_so_ext(), $
	                root_dir='/n/home08/dfink', subdirectory='test')


	c = fltarr(n, m)

	t1 = systime(1)
	for i=0L, niter-1 do retval = call_external(soname, 'matrixmult', m, n, k, cft, ddall, c)

	dt = systime(1)-t1
	print, (2LL*m*n*k*niter)/dt*1d-9, ' GFlops',  '  dt=', dt


	c = fltarr(m, n)

	cftt = transpose(cft)
	ddallt = transpose(ddall)
	t1 = systime(1)
	for i=0L, niter-1 do retval = call_external(soname, 'matrixmult', n, m, k, ddallt, cftt, c)

	dt = systime(1)-t1
	print, (2LL*m*n*k*niter)/dt*1d-9, ' GFlops',  '  dt=', dt




	return
end

# cff is 10x625
function dchi2dpsf(data,resid,x,y,flux,back,cff)
	# only play with 0-th order term in each pixel
	# so cff[1,:]
	T = Float32
	ndim = 10
	nstar = length(x)
	cff0   = copy(cff)
	model0 = zeros(T, size(resid))
	recon = Array{Float32,2}(undef, 625, nstar)

	image_model_eval!(model0, x, y, flux, back, cff0, recon)
	chi2 = sum((data-model0).^2)
	println("chi2:   ", chi2)
	# perturb params
	cf = copy(cff0)
	Δc = 0.001f0
	dpsf = zeros(T,ndim,625)
	for i = 1:ndim
		for j=1:625
			cf[i,j] += Δc
			model1 = zeros(T, size(resid))
			image_model_eval!(model1, x, y, flux, back, cf, recon)
			dchi2 = 0.0
			for k=1:length(model1)
				dchi2 += resid[k] * (model1[k]-model0[k])
			end
			dchi2 *= (2.0/Δc)
			dpsf[i,j] = dchi2
			cf[i,j] -= Δc
		end
	end
	return dpsf
end


function refine_psf(data,resid,x,y,flux,back,cff)
	ndim=10
	dpsf = dchi2dpsf(data,resid,x,y,flux,back,cff)
	# now we have our gradient.  Do a line search to get coeff
	ϵ  = 1e-10
	χ2 = 1e50
	γ  = 0.0
	for k=0:100
		γ = ϵ*1.2^k
		cf1 = copy(cff)
		cf1[1:ndim,:] += dpsf .* γ
		model1 = zeros(T, size(resid))
		image_model_eval!(model1, x, y, flux, back, cf1)
		prev_χ2 = χ2
		χ2 = sum((data-model1).^2)
		@printf("%3d %12.3e  %12.3f\n",k,γ,χ2)
		if χ2 > prev_χ2 break end
	end
	println("best γ", γ)
	cf1 = copy(cff)
	cf1[1:ndim,:] += dpsf .* γ

	return cf1
end

foo=readit()
data=foo[141:380,61:210]

resid,x,y,flux,back = mc(cff,data.*6)
cf1 = refine_psf(data.*6,resid,x,y,flux,back,cff)
resid,x,y,flux,back = mc(cf1,data.*6)
cf2 = refine_psf(data.*6,resid,x,y,flux,back,cf1)
resid,x,y,flux,back = mc(cf2,data.*6)

println(1)


pro getpsf, bpsf

# -------- Ask Aaron where these actually live
  infofile = '~/1497p015.1.info.fits'

  psf = mrdfits(infofile, 3)   # 325x325

  npix = 25
  fac  = 5
  nbig = npix*fac
  bar = [0.4, 0.2, 0., -0.2, -0.4]

  bpsf = fltarr(nbig, nbig)
  xind = (lindgen(npix, npix) mod npix)*fac
  yind = (lindgen(npix, npix)  /  npix)*fac
  for i=0, fac-1 do begin
     for j=0, fac-1 do begin
        bpsf[xind+i, yind+j] = (sshift2d(psf, [bar[i], bar[j]]))[150:174, 150:174]
     endfor
  endfor

  return
end



struct Point{T}
  x::T
  y::T
  z::T
end


using DataFrames

DataFrame(rand(5, 3), [:x, :y, :z])
ff=DataFrame(x = Int.(round(randn(10).*100)), y = randn(10))
