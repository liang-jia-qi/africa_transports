setwd("~/Documents/Research/Africa_Transports")

require(geosphere)
require(igraph)
require(rworldmap)
require(rworldxtra)
library(sf)

#### read data and params
{
AP <- read.csv("~/Documents/Research/Africa_Transports/input/AfricanAirpots.csv")
names(AP)[5] <- "y"
names(AP)[6] <- "x"
f <- AP$continent == "AF" & AP$iso_country != "ES"
f[is.na(f)] <- F
g <- AP$type == "large_airport" | AP$type == "medium_airport"
AP <- AP[f&g,]
n <- dim(AP)[1]
AP <- AP[order(AP$type, decreasing = F), ]
T0 <- 273000000 #### total passengers in 2025
gammaDist = 0.5 #### parameter for the impact of distance
}

# pop <- read.csv("input/airport_voronoi_population.csv")
# AP <- merge(AP, pop, by.x = "ident", by.y = "ident", all.x = TRUE)

#### generate synthetic population for each
{
  de <- (1:n)^(-.1) ### here, kappa for power law with coeff = -1
  P0 <- 100000000
  y0 <- rpois(n, lambda = de/sum(de)*P0)
  AP$pop <- y0
}


### vor tess
{
  library(sf)
  
  # 1) points -> sf (WGS84 raw input)
  pts <- st_as_sf(AP, coords = c("x", "y"), crs = 4326)
  
  # 2) EARLY projection (关键修正)
  # use equal-area projection to avoid Voronoi distortion
  pts_proj <- st_transform(pts, 8857)   # Africa Albers Equal Area (recommended)
  
  # 3) bounding box in projected space
  bb <- st_bbox(pts_proj)
  bb_poly <- st_as_sfc(bb)
  
  # small buffer in meters (now valid because projected CRS)
  pad <- 50000  # 50 km buffer, adjust if needed
  lim <- st_buffer(bb_poly, dist = pad)
  
  # 4) Voronoi tessellation (valid planar geometry now)
  v <- st_voronoi(st_union(pts_proj), envelope = lim)
  vor <- st_collection_extract(v, "POLYGON")
  
  vor_sf <- st_sf(
    id = seq_along(vor),
    geometry = vor,
    crs = st_crs(pts_proj)
  )
  
  # 5) back to geographic / raster CRS ONLY after geometry is correct
  
}

### calculate population
{
  library(terra)
  library(exactextractr)
  
  # 1. load population raster
  pop_raster <- rast("input/global_pop_2025_CN_1km_R2025A_UA_v1.tif")
  
  # 2. ensure CRS consistency
  vor_sf <- st_transform(vor_sf, st_crs(pop_raster))
  
  # 3. population extraction (safe sum)
  cat("Extracting population data...\n")
  vor_sf$pop <- exact_extract(pop_raster, vor_sf, 'sum')
  
  # 4. NA handling
  vor_sf$pop[is.na(vor_sf$pop)] <- 0
  
  # 5. map back to AP
  AP$pop <- vor_sf$pop
  
  # 6. save
  write.csv(AP,
            "~/Documents/Research/Africa_Transports/input/AP_with_Pop_2025.csv",
            row.names = FALSE)
}


#### estimate pairwise travel
#### function of income
#### remove short distance air travel
  AP <- read.csv("~/Documents/Research/Africa_Transports/input/AP_with_Pop_2025.csv")
  {
    AP$pop[is.na(AP$pop) | AP$pop <= 0] <- 100 # Floor for tiny airports
    D <- distm(AP[, c("x", "y")]) # Matrix of distances in meters
    # Apply a 100km (100,000m) cutoff: 
    # If distance is < 100km, make the distance effectively infinite so travel becomes 0
    D_weighted <- D
    D_weighted[D < 100000] <- Inf 
    P <- matrix(rep(AP$pop, times = n), ncol = n)
    # Using the weighted distance matrix
    OD.travel <- (P * t(P)) / (D_weighted + 1)^(gammaDist)
    # Clean up
    OD.travel[is.infinite(OD.travel)] <- 0
    diag(OD.travel) <- 0
    # Scale to your total passenger target T0
    OD.travel <- T0 * OD.travel / sum(OD.travel, na.rm = TRUE)
  }

#### assign journeys to routes
{
AP$closest <- AP$id[apply(D[, AP$type=="large_airport", drop=FALSE], 1, which.min) ]
OD <- rbind(
  expand.grid(
    OriginID      = AP$id[AP$type == "large_airport"],
    DestinationID = AP$id[AP$type == "large_airport"],
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  ),
  data.frame(
    OriginID      = AP$id[AP$type == "medium_airport"],
    DestinationID = AP$closest[AP$type == "medium_airport"],
    stringsAsFactors = FALSE
  )
)
OD <- as.data.frame(OD, stringsAsFactors = FALSE)
OD$Vol <- 0

directFlight <- 0
threeLegged <- 0

for(k in 1:n){for(j in 1:n){
  
  v <- OD.travel[k,j]
  ot <- AP$type[k]
  dt <- AP$type[j]

  if(ot == "large_airport" & dt == "large_airport"){ ### two large
    ToP <- which(OD$OriginID == AP$id[k] & OD$DestinationID == AP$id[j] )
    PoT <- which(OD$OriginID == AP$id[j] & OD$DestinationID == AP$id[k] )
    OD$Vol[ToP] <- OD$Vol[ToP] + v
    OD$Vol[PoT] <- OD$Vol[PoT] + v
    directFlight <- directFlight + v
  }
  
  if(ot == "medium_airport" & dt == "large_airport"){ ###medium to large
    if(AP$closest[k] == AP$id[j]){ ### the closest is the destionation
      ToP <- which(OD$OriginID == AP$id[k] & OD$DestinationID == AP$id[j] )
      PoT <- which(OD$OriginID == AP$id[j] & OD$DestinationID == AP$id[k] )
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
      directFlight <- directFlight + v
    } else {
      w <- AP$closest[k] #### there is a stop
      
      ToP <- which(OD$OriginID == AP$id[k] & OD$DestinationID == w)
      PoT <- which(OD$OriginID == w & OD$DestinationID == AP$id[k] )
      
      ToPs <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == AP$id[j] )
      
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
      
    }
  }

  if(dt == "medium_airport" & ot == "large_airport"){ ###large to medium
    if(AP$closest[j] == AP$id[k]){ ### the closest is the destionation
      ToP <- which(OD$OriginID == AP$id[k] & OD$DestinationID == AP$id[j] )
      PoT <- which(OD$OriginID == AP$id[j] & OD$DestinationID == AP$id[k] )
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
    } else {
      w <- AP$closest[j]
      
      ToP <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w)
      PoT <- which(OD$OriginID == w & OD$DestinationID == AP$id[j] )
      
      ToPs <- which(OD$OriginID == AP$id[k] & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == AP$id[k] )
      
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
      
    }
  }
  
  if(dt == "medium_airport" & ot == "medium_airport"){ ###medium to medium
    if(AP$closest[j] == AP$closest[k]){ ### the closest is the same
      w <- AP$closest[j] 
      ToP <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w)
      PoT <- which(OD$OriginID == w & OD$DestinationID == AP$id[j] )
      
      ToPs <- which(OD$OriginID == AP$id[k] & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == AP$id[k] )
      
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
    } else { #### the closest is not the same, so three journeys
      cat("three Journeys", "\n")
      w <- AP$closest[j]
      z <- AP$closest[k]
      
      ToP <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w )
      PoT <- which(OD$OriginID == w & OD$DestinationID == AP$id[j] )
      
      ToPs <- which(OD$OriginID == z & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == z)
      
      ToPt <- which(OD$OriginID == AP$id[k] & OD$DestinationID == z)
      PoTt <- which(OD$OriginID == z & OD$DestinationID == AP$id[k] )
      
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
      OD$Vol[ToPt] <- OD$Vol[ToPt] + v
      OD$Vol[PoTt] <- OD$Vol[PoTt] + v
      threeLegged <- threeLegged + v
    }
  }
}}

save(OD, file = "~/Documents/Research/Africa_Transports/OD20260306.RData")
}

#### assign journeys to routes (with 100km threshold)
{
n <- nrow(AP)

# --- distance constraint function ---
valid_link <- function(i, j) {
  return(D[i, j] >= 100000)
}

# --- recompute closest hub WITH threshold ---
D_hub <- D
D_hub[D_hub < 100000] <- Inf

AP$closest <- AP$id[
  apply(D_hub[, AP$type=="large_airport", drop=FALSE], 1, which.min)
]

# --- OD structure ---
OD <- rbind(
  expand.grid(
    OriginID      = AP$id[AP$type == "large_airport"],
    DestinationID = AP$id[AP$type == "large_airport"],
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  ),
  data.frame(
    OriginID      = AP$id[AP$type == "medium_airport"],
    DestinationID = AP$closest[AP$type == "medium_airport"],
    stringsAsFactors = FALSE
  )
)

OD <- as.data.frame(OD, stringsAsFactors = FALSE)
OD$Vol <- 0

directFlight <- 0
threeLegged <- 0

# --- main loop ---
for(k in 1:n){for(j in 1:n){
  if(k >= j) next
  v <- OD.travel[k,j]
  if(v == 0) next
  
  ot <- AP$type[k]
  dt <- AP$type[j]

  # =========================
  # large -> large
  # =========================
  if(ot == "large_airport" & dt == "large_airport"){
    
    if(!valid_link(k,j)) next
    
    ToP <- which(OD$OriginID == AP$id[k] & OD$DestinationID == AP$id[j])
    PoT <- which(OD$OriginID == AP$id[j] & OD$DestinationID == AP$id[k])
    
    OD$Vol[ToP] <- OD$Vol[ToP] + v
    OD$Vol[PoT] <- OD$Vol[PoT] + v
    
    directFlight <- directFlight + v
  }

  # =========================
  # medium -> large
  # =========================
  if(ot == "medium_airport" & dt == "large_airport"){
    
    j_idx <- j
    w <- AP$closest[k]
    w_idx <- which(AP$id == w)

    # direct
    if(AP$closest[k] == AP$id[j]){
      
      if(!valid_link(k, j_idx)) next
      
      ToP <- which(OD$OriginID == AP$id[k] & OD$DestinationID == AP$id[j])
      PoT <- which(OD$OriginID == AP$id[j] & OD$DestinationID == AP$id[k])
      
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
      
      directFlight <- directFlight + v
      
    } else {
      # via hub
      
      if(!valid_link(k, w_idx) || !valid_link(w_idx, j_idx)) next
      
      ToP  <- which(OD$OriginID == AP$id[k] & OD$DestinationID == w)
      PoT  <- which(OD$OriginID == w & OD$DestinationID == AP$id[k])
      
      ToPs <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == AP$id[j])
      
      OD$Vol[ToP]  <- OD$Vol[ToP]  + v
      OD$Vol[PoT]  <- OD$Vol[PoT]  + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
    }
  }

  # =========================
  # large -> medium
  # =========================
  if(dt == "medium_airport" & ot == "large_airport"){
    
    k_idx <- k
    w <- AP$closest[j]
    w_idx <- which(AP$id == w)

    if(AP$closest[j] == AP$id[k]){
      
      if(!valid_link(k_idx, j)) next
      
      ToP <- which(OD$OriginID == AP$id[k] & OD$DestinationID == AP$id[j])
      PoT <- which(OD$OriginID == AP$id[j] & OD$DestinationID == AP$id[k])
      
      OD$Vol[ToP] <- OD$Vol[ToP] + v
      OD$Vol[PoT] <- OD$Vol[PoT] + v
      
    } else {
      
      if(!valid_link(k_idx, w_idx) || !valid_link(w_idx, j)) next
      
      ToP  <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w)
      PoT  <- which(OD$OriginID == w & OD$DestinationID == AP$id[j])
      
      ToPs <- which(OD$OriginID == AP$id[k] & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == AP$id[k])
      
      OD$Vol[ToP]  <- OD$Vol[ToP]  + v
      OD$Vol[PoT]  <- OD$Vol[PoT]  + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
    }
  }

  # =========================
  # medium -> medium
  # =========================
  if(dt == "medium_airport" & ot == "medium_airport"){
    
    w <- AP$closest[j]
    z <- AP$closest[k]
    
    w_idx <- which(AP$id == w)
    z_idx <- which(AP$id == z)

    # same hub
    if(w == z){
      
      if(!valid_link(k, w_idx) || !valid_link(j, w_idx)) next
      
      ToP  <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w)
      PoT  <- which(OD$OriginID == w & OD$DestinationID == AP$id[j])
      
      ToPs <- which(OD$OriginID == AP$id[k] & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == AP$id[k])
      
      OD$Vol[ToP]  <- OD$Vol[ToP]  + v
      OD$Vol[PoT]  <- OD$Vol[PoT]  + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
      
    } else {
      # three-leg
      
      if(!valid_link(k, z_idx) ||
         !valid_link(z_idx, w_idx) ||
         !valid_link(w_idx, j)) next
      
      ToP  <- which(OD$OriginID == AP$id[j] & OD$DestinationID == w)
      PoT  <- which(OD$OriginID == w & OD$DestinationID == AP$id[j])
      
      ToPs <- which(OD$OriginID == z & OD$DestinationID == w)
      PoTs <- which(OD$OriginID == w & OD$DestinationID == z)
      
      ToPt <- which(OD$OriginID == AP$id[k] & OD$DestinationID == z)
      PoTt <- which(OD$OriginID == z & OD$DestinationID == AP$id[k])
      
      OD$Vol[ToP]  <- OD$Vol[ToP]  + v
      OD$Vol[PoT]  <- OD$Vol[PoT]  + v
      OD$Vol[ToPs] <- OD$Vol[ToPs] + v
      OD$Vol[PoTs] <- OD$Vol[PoTs] + v
      OD$Vol[ToPt] <- OD$Vol[ToPt] + v
      OD$Vol[PoTt] <- OD$Vol[PoTt] + v
      
      threeLegged <- threeLegged + v
    }
  }

}}

save(OD, file = "~/Documents/Research/Africa_Transports/OD20260306.RData")
}

#### figures
#### base map with airports
{
  pdf(width = 8, height = 7,
      file = "~/Documents/Research/Africa_Transports/Figures/BaseAirports20260306.PDF")
  data(countriesHigh)
  shape <- st_read(dsn = "~/Documents/Research/Africa_Transports/input/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
  shape <- st_read(dsn = "~/Documents/Research/Africa_Transports/input/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
  #### map with most frequent modal share
  par(mar = c(0,0,0,0))
  plot(AP$x, AP$y, col = NA, asp = 1)
  polygon(c(-1000, 1000, 1000,-1000), 
          c(-100,-100,100,100),
          col = "lightcyan1")
  plot(shape, add = T, 
       col = rgb(1,1,1,0.5),
       lwd = 0.2,
       border = "gray90")
  plot(countriesHigh,
       col = "NA",
       border = 1,
       lwd = .5,
       add = T)
  
  MAP <- AP[AP$type == "medium_airport", ]
  LAP <- AP[AP$type == "large_airport", ]
  
  points(MAP$x, MAP$y, pch = 21, bg = "gold", 
         cex = 1)
  points(LAP$x, LAP$y, pch = 21, bg = "tomato", 
         cex = 1.5)
  dev.off()
}

#### vor polygon with airports
{
pdf(width = 8, height = 7,
  file = "~/Documents/Research/Africa_Transports/Figures/VoronoiAirports20260306.PDF")
data(countriesHigh)
shape <- st_read(dsn = "~/Documents/Research/Africa_Transports/input/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
shape <- st_read(dsn = "~/Documents/Research/Africa_Transports/input/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
#### map with most frequent modal share
par(mar = c(0,0,0,0))
plot(AP$x, AP$y, col = NA, asp = 1)
polygon(c(-1000, 1000, 1000,-1000), 
        c(-100,-100,100,100),
        col = "lightcyan1")
plot(vor_sf$geometry,
     col = adjustcolor("deepskyblue3", alpha.f = 0.35),
     border = adjustcolor("grey20", alpha.f = 0.5), add = T)
plot(shape, add = T, 
     col = rgb(1,1,1,0.5),
     lwd = 0.2,
     border = "gray90")
plot(countriesHigh,
     col = "NA",
     border = 1,
     lwd = .5,
     add = T)

points(MAP$x, MAP$y, pch = 21, bg = "gold", 
       cex = 1)
points(LAP$x, LAP$y, pch = 21, bg = "tomato", 
       cex = 1.5)
dev.off()
}

#### map with top routes
{
pdf(width = 8, height = 7,
      file = "~/Documents/Research/Africa_Transports/Figures/SimTravelAirports20260306.PDF")
data(countriesHigh)
shape <- st_read(dsn = "~/Documents/Research/Africa_Transports/input/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
shape <- st_read(dsn = "~/Documents/Research/Africa_Transports/input/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
#### map with most frequent modal share
OD <- OD[order(OD$Vol, decreasing = T), ]
par(mar = c(0,0,0,0))
plot(AP$x, AP$y, col = NA, asp = 1)
polygon(c(-1000, 1000, 1000,-1000), 
        c(-100,-100,100,100),
        col = "lightcyan1")
plot(countriesHigh,
     col = "NA",
     border = 1,
     lwd = .5,
     add = T)
LAP <- AP[AP$type == "large_airport",]
MAP <- AP[AP$type != "large_airport",]
# map IDs -> coordinates (vectorized)
toF <- c(0.01, 0.05, 0.1, 0.2, 0.5)
cols <- c("gray80",
          "gray60",
          "gray40",
          "gray20",
          "gray0"
          )
for(tk in 1:5){
TopR <- OD$Vol > max(OD$Vol)*toF[tk]
io <- match(OD$OriginID[TopR], AP$id)
id <- match(OD$DestinationID[TopR], AP$id)
lwd <- log(OD$Vol[TopR]) / log(diff(range(OD$Vol)))

segments(x0=AP$x[io], y0=AP$y[io],
         x1=AP$x[id], y1=AP$y[id],
         col = cols[tk],
         lwd=tk)
}

ntk <- 25000
for(tk in ntk:1){
  io <- match(OD$OriginID[tk], AP$id)
  id <- match(OD$DestinationID[tk], AP$id)
  lwd <- log(OD$Vol[tk]) / log(diff(range(OD$Vol)))
  
  segments(x0=AP$x[io], y0=AP$y[io],
           x1=AP$x[id], y1=AP$y[id],
           col = "gray20",
           lwd=(ntk-tk+1)/ntk)
}
points(MAP$x, MAP$y, pch = 21, bg = "gold", 
       cex = 1)
points(LAP$x, LAP$y, pch = 21, bg = "tomato", 
       cex = 1.5)
dev.off()
}


{
# 1. Sort the OD dataframe by Volume
OD_sorted <- OD[order(OD$Vol, decreasing = TRUE), ]

# 2. Match IDs to Airport Names for readability
OD_sorted$OriginName <- AP$name[match(OD_sorted$OriginID, AP$id)]
OD_sorted$DestName   <- AP$name[match(OD_sorted$DestinationID, AP$id)]

# 3. View the Top 20
top_20 <- head(OD_sorted[, c("OriginName", "DestName", "Vol")], 20)
print(top_20)
}
