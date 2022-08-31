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
      Seg opacity <input type="range" min="0" max="100" value="30" class="slider" id="drawOpacity">
	</div>
      
<script>
   var slider = document.getElementById("gammaSlider");
	slider.oninput = function() {
       nv.setGamma(this.value * 0.01)
	}
	document.getElementById("drawOpacity").addEventListener("change", doDrawOpacity);
	function doDrawOpacity(){
		nv.setDrawOpacity(this.value * 0.01);
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
        colorMap: "winter",
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
 nv.setSliceType(nv.sliceTypeMultiPlanar) // press the "v" key to cycle through views
</script>


<!--url: "./results/MALC2012_1000_3_256iso_predicted_volume.nii.gz",
-->






