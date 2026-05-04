setwd("~/Documents/Research/Africa_Transports")

require(igraph)

#### load data
{
  Nodes <- read.csv("input/AfricaNetworkNodes.csv")
  Edges <- read.csv("input/AfricaNetworkEdges.csv")
  AfP <- read.csv("input/Africapolis_2050.csv")
  ToM <- AfP[c("Agglomeration_ID", "Population_2050")]
  names(ToM)[1] <- "name"
  names(ToM)[2] <- "Pop2050"
  Nodes <- merge(Nodes, ToM, all.x = T)
  Nodes$Pop2050[is.na(Nodes$Pop2050)] <- 0
  counL <- data.frame(ISO3 = unique(Nodes$ISO3),
                      ISOCode = 1:length(unique(Nodes$ISO3)))
  Nodes <- merge(Nodes, counL)
  Nodes <- Nodes[, c(2:6, 1, 7:10)]
  G <- graph_from_data_frame(Edges,
                             directed = F, 
                             vertices = Nodes)
  cities <- Nodes[Nodes$Pop2015>0, ]
  citiesFilt <- which(Nodes$Pop2015>0)
  DN <- distances(G, 
                  weights = edge.attributes(G)$l)
  DN <- DN[citiesFilt,]
  DN <- DN[, citiesFilt]
}

#### get gravity 
{
#### 2015
{
  NetworkB <- rep(0, vcount(G))
  nf <- which(vertex.attributes(G)$Pop2015>0) #filter cities
  nCities <- length(nf)
  CitiesP <- vertex.attributes(G)$Pop2015
  PopM <- matrix(rep(CitiesP, length(CitiesP)),
                 ncol = length(CitiesP))/1000
  NetD <- distances(G, 
                    #                  v = nf,
                    #                  to = nf,
                    weights = edge.attributes(G)$time)
  
  GravM <- PopM * t(PopM) / (NetD^2.8 + 1)
  GravM[upper.tri(GravM, diag = T)] <- 0
}

#### get gravity 2050
{
  NetworkB <- rep(0, vcount(G))
  CitiesP <- vertex.attributes(G)$Pop2050
  PopM <- matrix(rep(CitiesP, length(CitiesP)),
                 ncol = length(CitiesP))/1000
  GravM50 <- PopM * t(PopM) / (NetD^2.8 + 1)
  GravM50[upper.tri(GravM50, diag = T)] <- 0
  rows <- row(GravM50)
  cols <- col(GravM50)
  GM <- data.frame(from = as.vector(cols), 
                   to = as.vector(rows), 
                   GravM = as.vector(GravM),
                   GravM50 = as.vector(GravM50))
  u <- order(GM$GravM, decreasing = T)
  GM <- GM[u,]
  save(GM, file = "output/GravityPairs_pop2020_2050_20260317.RData")

}
}

#### get edge beweeness
#### now the dataframe GM has gravity and it is sorted
{
  nsteps <- 25000
  ToP <- (1:25)*2000
  # edge.attributes(G)$Between <- rep(0, ecount(G))
  # edge.attributes(G)$Between50 <- rep(0, ecount(G))
  Between  <- numeric(ecount(G))
  Between50 <- numeric(ecount(G))
  for(k in (nsteps+1):(2*nsteps)){
    
    if(k %% 200 == 0){
      cat(100*k/nsteps, "%\n")
      gc()
    }
    
    ASP <- shortest_paths(
      G,
      from = GM$from[k],
      to = GM$to[k],
      weights = edge.attributes(G)$time,
      output = "epath"
    )
    
    edge_ids <- as.integer(ASP$epath[[1]])
    
    if(length(edge_ids) == 0) next
    
    Between[edge_ids]  <- Between[edge_ids]  + GM$GravM[k]
    Between50[edge_ids] <- Between50[edge_ids] + GM$GravM50[k]
  }
  E(G)$Between  <- Between
  E(G)$Between50 <- Between50
  
  for(k in (nsteps+1):(2*nsteps)){
    if(k %in% ToP){cat(100*k/nsteps, " % \n")}
      ASP <- shortest_paths(G, 
                            from = GM$from[k],
                            to = GM$to[k],
                            weights = edge.attributes(G)$time)
      
      
      # Extract the vertex ids
      vpath_ids <- as.integer(ASP$vpath[[1]])
      
      # Skip if no path exists
      if (length(vpath_ids) < 2) next
      
      # Get pairs of consecutive vertices
      vertex_pairs <- embed(rev(vpath_ids), 2)
      
      # Find the edge ids
      edge_ids <- apply(vertex_pairs, 1, function(x) get_edge_ids(G, vp = x))
      
      # Increment Between count for these edges
      E(G)$Between[edge_ids] <- E(G)$Between[edge_ids] + GM$GravM[k]
      E(G)$Between50[edge_ids] <- E(G)$Between50[edge_ids] + GM$GravM50[k]
    
  }


save(G, file = "output/NetworkEdgeBetween_pop2020_2050_20250715.RData")
}

##############
## ANALYSIS
##############

####load data
{
load(file = "output/NetworkEdgeBetween_pop2020_2050_20250715.RData")
load(file = "output/GravityPairs_pop2020_2050_20250715.RData")
EdgeB <- as_data_frame(G, what = "edges")
f <- EdgeB$Between > 0
EdgeB <- EdgeB[f, ]
u <- order(GM$GravM50, decreasing = T)
GM <- GM[u, ]
}

##### increasing travel
### sum(GM$GravM50)/sum(GM$GravM)

#### FIGURES
#### network with all edges with width of the congestion 2015
{
  pdf("Figures/Congestion2015.pdf", width = 25, height = 25)
colN <- c("#FFC107", "#7A98D6", "tomato", "#00CC99", "#CC90C1")
# Create a named vector mapping continent to color
continent_colors <- setNames(colN[seq_along(unique(V(G)$Region))], 
                             unique(V(G)$Region))
node_colors <- continent_colors[V(G)$Region]

L <- cbind(Nodes$x, Nodes$y)
par(mar = c(0,0,0,0))
plot(G, 
     layout = L, 
     edge.color = "gray40",
     edge.width = 2*log(E(G)$Between+1)/1,
     vertex.frame.color = adjustcolor("black", alpha.f = 1),
     verte.frame.width = 0.01,
     vertex.size = log(V(G)$Pop2015+1)/16,
     vertex.color = node_colors,
     vertex.label = NA)
dev.off()
}

#### network with all edges with width of the congestion 2050
{
  pdf("figures/Congestion2050.pdf", width = 25, height = 25)
  colN <- c("#FFC107", "#7A98D6", "tomato", "#00CC99", "#CC90C1")
  # Create a named vector mapping continent to color
  continent_colors <- setNames(colN[seq_along(unique(V(G)$Region))], 
                               unique(V(G)$Region))
  node_colors <- continent_colors[V(G)$Region]
  
  L <- cbind(Nodes$x, Nodes$y)
  par(mar = c(0,0,0,0))
  plot(G, 
       layout = L, 
       edge.color = "gray40",
       edge.width = 2*log(E(G)$Between50+1)/1,
       vertex.frame.color = adjustcolor("black", alpha.f = 1),
       verte.frame.width = 0.01,
       vertex.size = log(V(G)$Pop2050+1)/16,
       vertex.color = node_colors,
       vertex.label = NA)
  dev.off()
}

#### road profile for any two cities with menu
{
  # Specify origin and destination nodes
  origin_node <- which(Nodes$agglosName == "Kano")
  destination_node <- which(Nodes$agglosName == "Abuja")
  destination_node <- which(Nodes$agglosName == "Lagos")
  
  origin_node <- which(Nodes$agglosName == "Lagos")
  destination_node <- which(Nodes$agglosName == "Abidjan")
  
  
  # Compute shortest path using edge length 'l'
  sp_result <- shortest_paths(
    G,
    from = origin_node,
    to = destination_node,
    weights = E(G)$l,
    output = "vpath"
  )
  sp_nodes <- as.integer(sp_result$vpath[[1]])
  
  # Prepare data frame of edges along the path
  n_edges <- length(sp_nodes) - 1
  x_start <- numeric(n_edges)
  x_end <- numeric(n_edges)
  y_bottom <- numeric(n_edges)
  y_top <- numeric(n_edges)
  
  cum_dist <- 0
  
  for (i in 1:n_edges) {
    from_node <- sp_nodes[i]
    to_node <- sp_nodes[i + 1]
    edge_id <- get.edge.ids(G, c(from_node, to_node))
    length_l <- E(G)$l[edge_id]
    bet_val <- E(G)$Between[edge_id]
    
    x_start[i] <- cum_dist
    x_end[i] <- cum_dist + length_l
    y_bottom[i] <- 0
    y_top[i] <- bet_val
    
    cum_dist <- cum_dist + length_l
  }
  
  # Set up empty plot
  plot(
    NULL,
    xlim = c(0, max(x_end)),
    ylim = c(0, max(y_top) * 1.1),
    xlab = "travelled dstance",
    ylab = "BET",
    main = paste("Path Profile: Node", origin_node, "to", destination_node)
  )
  
  # Draw rectangles
  for (i in 1:n_edges) {
    rect(
      xleft = x_start[i],
      ybottom = y_bottom[i],
      xright = x_end[i],
      ytop = y_top[i],
      col = "lightblue",
      border = "black"
    )
  }
  
  # Optionally add grid
  grid()
}

#### infographic for Lagos 2015
{
  pdf("Figures/LagosCongestion2015.pdf", width = 5, height = 5)
  par(mfrow = c(5,1), mar = c(0,0,0,0), oma = c(0,0,0,0))
  origin_node <-  which(Nodes$agglosName == "Lagos") ### Lagos. which(Nodes$agglosName == "Kano")
  ToDests <- c( which(Nodes$agglosName == "Kano"), #KANO
                which(Nodes$agglosName == "Abuja"), #abuja
                which(Nodes$agglosName == "Onitsha"), ### onitsha
                which(Nodes$agglosName == "Accra"), ### Accra
                which(Nodes$agglosName == "Abidjan") ### abidjan
  )
  CityN <- c("Kano", "Abuja", "Onitsha", "Accra", "Abidjan")
  for(Dest in 1:5){

  destination_node <- ToDests[Dest]
  
  
  # Compute shortest path using edge length 'l'
  sp_result <- shortest_paths(
    G,
    from = origin_node,
    to = destination_node,
    weights = E(G)$l,
    output = "vpath"
  )
  sp_nodes <- as.integer(sp_result$vpath[[1]])
  
  # Prepare data frame of edges along the path
  n_edges <- length(sp_nodes) - 1
  x_start <- numeric(n_edges)
  x_end <- numeric(n_edges)
  y_bottom <- numeric(n_edges)
  y_top <- numeric(n_edges)
  
  cum_dist <- 0
  
  for (i in 1:n_edges) {
    from_node <- sp_nodes[i]
    to_node <- sp_nodes[i + 1]
    edge_id <- get.edge.ids(G, c(from_node, to_node))
    length_l <- E(G)$l[edge_id]
    bet_val <- E(G)$Between[edge_id]
    
    x_start[i] <- cum_dist
    x_end[i] <- cum_dist + length_l
    y_bottom[i] <- 0
    y_top[i] <- bet_val
    
    cum_dist <- cum_dist + length_l
  }
  
  ymax = 15000
  # Set up empty plot
  plot(
    NULL,
    xlim = c(0, 900),
    ylim = c(1, ymax * 1.1),
    xlab = "",
    xaxt = "n", yaxt = "n",
    log = "y",
    ylab = "")
  
  # Draw rectangles
  for (i in 1:n_edges) {
    rect(
      xleft = x_start[i],
      ybottom = y_bottom[i]+1,
      xright = x_end[i],
      ytop = y_top[i]+1,
      col = colN[Dest],
      border = "NA"
    )
    text(cum_dist, 0.8*ymax, CityN[Dest], adj = 1)
  }
  }
  dev.off()
}

#### infographic for Lagos 2050
{
  pdf("Figures/LagosCongestion2050.pdf", width = 5, height = 5)
  par(mfrow = c(5,1), mar = c(0,0,0,0), oma = c(0,0,0,0))
  origin_node <-  which(Nodes$agglosName == "Lagos") ### Lagos. which(Nodes$agglosName == "Kano")
  ToDests <- c( which(Nodes$agglosName == "Kano"), #KANO
                which(Nodes$agglosName == "Abuja"), #abuja
                which(Nodes$agglosName == "Onitsha"), ### onitsha
                which(Nodes$agglosName == "Accra"), ### Accra
                which(Nodes$agglosName == "Abidjan") ### abidjan
  )
  CityN <- c("Kano", "Abuja", "Onitsha", "Accra", "Abidjan")
  for(Dest in 1:5){
    
    destination_node <- ToDests[Dest]
    
    
    # Compute shortest path using edge length 'l'
    sp_result <- shortest_paths(
      G,
      from = origin_node,
      to = destination_node,
      weights = E(G)$l,
      output = "vpath"
    )
    sp_nodes <- as.integer(sp_result$vpath[[1]])
    
    # Prepare data frame of edges along the path
    n_edges <- length(sp_nodes) - 1
    x_start <- numeric(n_edges)
    x_end <- numeric(n_edges)
    y_bottom <- numeric(n_edges)
    y_top <- numeric(n_edges)
    
    cum_dist <- 0
    
    for (i in 1:n_edges) {
      from_node <- sp_nodes[i]
      to_node <- sp_nodes[i + 1]
      edge_id <- get.edge.ids(G, c(from_node, to_node))
      length_l <- E(G)$l[edge_id]
      bet_val <- E(G)$Between50[edge_id]
      
      x_start[i] <- cum_dist
      x_end[i] <- cum_dist + length_l
      y_bottom[i] <- 0
      y_top[i] <- bet_val
      
      cum_dist <- cum_dist + length_l
    }
    
    ymax = 15000
    # Set up empty plot
    plot(
      NULL,
      xlim = c(0, 900),
      ylim = c(1, ymax * 1.1),
      xlab = "",
      xaxt = "n", yaxt = "n",
      log = "y",
      ylab = "")
    
    # Draw rectangles
    for (i in 1:n_edges) {
      rect(
        xleft = x_start[i],
        ybottom = y_bottom[i]+1,
        xright = x_end[i],
        ytop = y_top[i]+1,
        col = colN[Dest],
        border = "NA"
      )
      text(cum_dist, 0.8*ymax, CityN[Dest], adj = 1)
    }
  }
  dev.off()
}

#### infographic for Lagos 2015 - 2050
{
  pdf("Figures/LagosCongestion2015-2050.pdf", width = 5, height = 5)
  par(mfrow = c(5,1), mar = c(0,0,0,0), oma = c(0,0,0,0))
  origin_node <-  which(Nodes$agglosName == "Lagos") ### Lagos. which(Nodes$agglosName == "Kano")
  ToDests <- c( which(Nodes$agglosName == "Kano"), #KANO
                which(Nodes$agglosName == "Abuja"), #abuja
                which(Nodes$agglosName == "Onitsha"), ### onitsha
                which(Nodes$agglosName == "Accra"), ### Accra
                which(Nodes$agglosName == "Abidjan") ### abidjan
  )
  CityN <- c("Kano", "Abuja", "Onitsha", "Accra", "Abidjan")
  for(Dest in 1:5){
    # Set up empty plot
    plot(
      NULL,
      xlim = c(0, 900),
      ylim = c(1, ymax * 1.1),
      xlab = "",
      xaxt = "n", yaxt = "n",
      log = "y",
      ylab = "")
    
    
    destination_node <- ToDests[Dest]
    
    # Compute shortest path using edge length 'l'
    sp_result <- shortest_paths(
      G,
      from = origin_node,
      to = destination_node,
      weights = E(G)$l,
      output = "vpath"
    )
    sp_nodes <- as.integer(sp_result$vpath[[1]])
    
    # Prepare data frame of edges along the path
    n_edges <- length(sp_nodes) - 1
    x_start <- numeric(n_edges)
    x_end <- numeric(n_edges)
    y_bottom <- numeric(n_edges)
    y_top <- numeric(n_edges)
    
    cum_dist <- 0
    
    for (i in 1:n_edges) {
      from_node <- sp_nodes[i]
      to_node <- sp_nodes[i + 1]
      edge_id <- get.edge.ids(G, c(from_node, to_node))
      length_l <- E(G)$l[edge_id]
      bet_val <- E(G)$Between50[edge_id]
      
      x_start[i] <- cum_dist
      x_end[i] <- cum_dist + length_l
      y_bottom[i] <- 0
      y_top[i] <- bet_val
      
      cum_dist <- cum_dist + length_l
    }
    
    ymax = 15000

    
    # Draw rectangles
    for (i in 1:n_edges) {
      rect(
        xleft = x_start[i],
        ybottom = y_bottom[i]+1,
        xright = x_end[i],
        ytop = y_top[i]+1,
        col = colN[Dest],
        border = "NA"
      )
      text(cum_dist, 0.8*ymax, CityN[Dest], adj = 1)
    }
    
    
    
    cum_dist <- 0
    
    for (i in 1:n_edges) {
      from_node <- sp_nodes[i]
      to_node <- sp_nodes[i + 1]
      edge_id <- get.edge.ids(G, c(from_node, to_node))
      length_l <- E(G)$l[edge_id]
      bet_val <- E(G)$Between[edge_id]
      
      x_start[i] <- cum_dist
      x_end[i] <- cum_dist + length_l
      y_bottom[i] <- 0
      y_top[i] <- bet_val
      
      cum_dist <- cum_dist + length_l
    }
    
    ymax = 15000
    
    
    # Draw rectangles
    for (i in 1:n_edges) {
      rect(
        xleft = x_start[i],
        ybottom = y_bottom[i]+1,
        xright = x_end[i],
        ytop = y_top[i]+1,
        col = "white",
        border = "NA"
      )
      text(cum_dist, 0.8*ymax, CityN[Dest], adj = 1)
    }
    
  }
  dev.off()
}

#### congestion in 2015 and 2050 by road
{
plot(EdgeB $Between+1, 
     EdgeB $Between50+1,
     log = "xy",
     xlab = "congestion in 2015",
     ylab = "congestion in 2050")
M <- 1000000000
points(c(1,M), c(1,M), type = "l", col = 2)
}
