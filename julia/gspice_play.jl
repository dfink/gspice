#using Revise
using Plots
using BenchmarkTools
using LinearAlgebra
using Statistics
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
