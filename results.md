---
layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

[<-- main page](https://rocknroll87q.github.io/LOD-Brain/)

<hr>

## First result

<script src="https://unpkg.com/@niivue/niivue@0.29.0/dist/niivue.umd.js"></script>


<section>

  <div class="slidecontainer">
    seg opacity<input type="range" min="1" max="255" value="75" class="slider" id="alphaSlider">
  </div>

  
  <div id="demo1" style="width:1000px; height:1000px;">
    <canvas id="gl1" height=640 width=640>
    </canvas>
  </div>
    
</section>



<script>
 var volumeList1 = [
   // first item is background image
     {
       url: "./results/MALC2012_1000_3_256iso_t1.nii.gz",
       colorMap: "gray",
     },
     {
       url: "./results/MALC2012_1000_3_256iso_predicted_volume.nii.gz",
       colorMap: "random",
       opacity: 0.3,
     },
    ] 
  function handleLocationChange(data){
    document.getElementById('location').innerHTML = data.xy
  }
  var nv1 = new niivue.Niivue({onLocationChange:handleLocationChange})
  nv1.attachTo('gl1')
  nv1.loadVolumes(volumeList1)
  //nv1.setSliceType(nv1.sliceTypeRender)
  var slider = document.getElementById("alphaSlider");
  slider.oninput = function() {
    nv1.setOpacity (1, this.value / 255);
  }

</script>

