layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

[<-- main page](https://rocknroll87q.github.io/LOD-Brain/)

<hr>

## First result


<script src="https://unpkg.com/@niivue/niivue@0.29.0/dist/niivue.umd.js"></script>
  


<canvas id="gl" height=600></canvas>
  <div class="slidecontainer">
    T1 gamma  <input type="range" min="10" max="400" value="100" class="slider" id="gammaSlider">
  </div>
  <div class="slidecontainer">
    seg opacity<input type="range" min="1" max="255" value="77" class="slider" id="alphaSlider">
  </div>
  <input type="checkbox" id="check1" name="check1" unchecked>
  <label for="check1">contours only</label>
        
<script>
   var slider = document.getElementById("gammaSlider");
	slider.oninput = function() {
       nv.setGamma(this.value * 0.01)
	}
   slider.oninput = function() {
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


<!--url: "./results/MALC2012_1000_3_256iso_predicted_volume.nii.gz",
-->






