https://discourse.julialang.org/t/extract-type-name-only-from-parametric-type/14188/23
```julia
for type in subtypes(AbstractArray)
  @eval getname(x::$type) = $type
end
```

https://discourse.julialang.org/t/stripping-parameter-from-parametric-types/8293/16
```julia
getname(type::Type) = Base.typename(type).wrapper
```

https://docs.julialang.org/en/v1/manual/methods/#Extracting-the-type-parameter-from-a-super-type

# Overloads needed for `AbstractArray` type
```julia
parameter(::Type{<:AbstractArray{P1}}, ::Position{1}) where {P1} = P1
parameter(::Type{<:AbstractArray{<:Any,P2}}, ::Position{2}) where {P2} = P2

set_parameter(::Type{<:AbstractArray}, ::Position{1}, P1) = AbstractArray{P1}
set_parameter(::Type{<:AbstractArray{<:Any,P2}}, ::Position{1}, P1) where {P2} = AbstractArray{P1,P2}
set_parameter(::Type{<:AbstractArray}, ::Position{2}, P2) = AbstractArray{<:Any,P2}
set_parameter(::Type{<:AbstractArray{P1}}, ::Position{2}, P2) where {P1} = AbstractArray{P1,P2}

default_parameter(::Type{<:AbstractArray}, ::Position{1}) = Float64
default_parameter(::Type{<:AbstractArray}, ::Position{2}) = 1

nparameters(::Type{<:AbstractArray}) = Val(2)
```

https://github.com/ITensor/ITensors.jl/pull/1213/files#r1431708585

# Create generic set_eltype and set_ndims functions which can be defined on
# Array structures and use the `set_parameter`
```julia
set_eltype(T::Type, elt::Type) = Error("set_eltype is not defined for datatype $T")
set_eltype(T::Type{<:Array}, elt::Type) = set_parameter(T, Position{1}(), elt)
```