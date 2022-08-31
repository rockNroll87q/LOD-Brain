---
layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

[<-- main page](https://rocknroll87q.github.io/LOD-Brain/)

<hr>

## First result

<!--
<script src="https://unpkg.com/@niivue/niivue@0.29.0/dist/niivue.umd.js"></script>
  


<canvas id="gl" height=600></canvas>
  <div class="slidecontainer">
    T1 gamma  <input type="range" min="10" max="400" value="100" class="slider" id="gammaSlider">
  </div>
  <div class="slidecontainer">
    seg opacity<input type="range" min="1" max="255" value="77" class="slider" id="alphaSlider">
  </div>
  <div class="contours">
    <input type="checkbox" id="check1" name="check1" unchecked>
    <label for="check1">contours only</label>
  </div>
  
<script>
   var slider = document.getElementById("gammaSlider");
	slider.oninput = function() {
       nv.setGamma(this.value * 0.01)
	}
	var slider1 = document.getElementById("alphaSlider");
   slider1.oninput = function() {
		nv.setOpacity (1, this.value / 255);
	}
	
  var volumeList = [
    // first object in array is background image
      {
        url: "./results/MALC2012_1000_3_256iso_t1.nii.gz",
        volume: {hdr: null, img: null},
        name: "MALC2012_1000_3",
        colorMap: "gray",
        opacity: 1,
        visible: true,        
      },
	  {
        url: "./results/MALC2012_1000_3_256iso_predicted_volume.nii.gz",
        volume: {hdr: null, img: null},
        name: "seg",
        colorMap: "random",
        opacity: 0.3,
        visible: true,        
      }
   ]

 // Niivue will adjust the canvas to 100% of its parent container's size 
 // the parent element can be any size you want (small or large)
 var nv = new niivue.Niivue({
 	backColor: [0., 0., 0., 1],
 	})
 nv.attachTo('gl') // the canvas element id
 nv.loadVolumes(volumeList)
nv.setSliceType(nv.sliceTypeMultiPlanar)
 
	document.getElementById("check1").addEventListener("change", doCheckClick);
	function doCheckClick() {
	    nv.setAtlasOutline(this.checked)
	}
</script>
-->

<script src="https://unpkg.com/@niivue/niivue@0.29.0/dist/niivue.umd.js"></script>


<section>
  <div class="slidecontainer">
  <input type="checkbox" id="check1" name="check1" unchecked>
  <label for="check1">outline</label>
    </div>
  <div class="slidecontainer">
    seg opacity<input type="range" min="1" max="255" value="75" class="slider" id="alphaSlider">
  </div>
  <div class="slidecontainer">
    T1 gamma  <input type="range" min="10" max="400" value="100" class="slider" id="gammaSlider">
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
     },
    ] 
  function handleLocationChange(data){
    document.getElementById('location').innerHTML = data.xy
  }
  var nv1 = new niivue.Niivue({onLocationChange:handleLocationChange})
  nv1.attachTo('gl1')
  nv1.loadVolumes(volumeList1)
  //nv1.setSliceType(nv1.sliceTypeRender)
  document.getElementById("check1").addEventListener("change", doCheckClick);
  function doCheckClick() {
    nv1.setAtlasOutline(this.checked)
  }
  var slider = document.getElementById("alphaSlider");
  slider.oninput = function() {
    nv1.setOpacity (1, this.value / 255);
  }
   var slider2 = document.getElementById("gammaSlider");
	slider2.oninput = function() {
       nv1.setGamma(this.value * 0.01)
	}  
  let query = window.location.search
  nv1.on('location', (data) => {
    // data is an object with properties: {mm: [N N N], vox: [N N N], frac: [N N N]}
    //document.getElementById('location').innerHTML = 'voxel location: ' + data.vox + ' ' + data.values
    document.getElementById('location').innerHTML = data.xy
  })
</script>


