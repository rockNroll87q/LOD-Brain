---
layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

[<-- main page](https://rocknroll87q.github.io/LOD-Brain/)

<hr>

## Segmentation result

We select some volumes with the worst numerical result (max one for dataset), and display FreeSurfer and LOD-Brain segmentation masks overlayed to the correspondent T1w image. Despite low performance, the segmentation mask returned by LOD-Brain shown less errors with respect to Freesurfer, which is indicative of low-quality ground-truth masks.

* Click the buttons to load test images.
* Press `v` to toggle between views.
* Press `c` for interactive clip plane.
* Move the crosshairs: left mouse click and drag.
* Slice scrolling: mouse or touch pad scroll up and down.
* Set opacity and gamma with sliders below.




<script src="https://unpkg.com/@niivue/niivue@0.29.0/dist/niivue.umd.js"></script>

<section>

  <div class="header_showing_results">
	  <p>Test images:</p>
	  <p id="images"></p>
	  
  </div>
  <div class="header_showing_results">
    seg opacity<input type="range" min="1" max="255" value="75" class="slider" id="alphaSlider">
  </div>
  <div class="header_showing_results">
    T1 gamma  <input type="range" min="10" max="400" value="100" class="slider" id="gammaSlider">
  </div> 
  <div class="header_showing_results">  
  <label for="mask_to_show">Segmentation mask</label>
	<select name="myMask" id="mask_to_show">
	  <option value="LOD-Brain">LOD-Brain</option>
	  <option value="FreeSurfer">FreeSurfer</option>
	</select>
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
       url: "./results/AOMIC_T1w.nii.gz",
       colorMap: "gray",
     },
     {
       url: "./results/AOMIC_pred.nii.gz",
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
  nv1.setHighResolutionCapable(this.checked);
  //nv1.setSliceType(nv1.sliceTypeRender)
  var slider = document.getElementById("alphaSlider");
  slider.oninput = function() {
    nv1.setOpacity (1, this.value / 255);
  }
   var slider2 = document.getElementById("gammaSlider");
	slider2.oninput = function() {
       nv1.setGamma(this.value * 0.01)
	}  
	
	imgs = ["AOMIC", "EDSD", "HCP", "IBSR", "IXI", "MRBrainS", "MindBoggle101", "OASIS3"]
	imgEl = document.getElementById('images')
	var maskToShow = document.getElementById("mask_to_show")
	for (let i=0; i<imgs.length; i++) {
		let btn = document.createElement("button")
		btn.innerHTML = imgs[i]
		btn.onclick = function() {
		  let root = './results/'
		  let img_t1 = root + imgs[i] + '_T1w.nii.gz'
		  let img_mask = root + imgs[i]  
		  maskToShow.onchange = function() {
		    switch(document.getElementById("mask_to_show").value) {
		    		case "LOD-Brain":
						img_mask = img_mask + '_pred.nii.gz'
					case "FreeSurfer":
						img_mask = img_mask + '_GT.nii.gz'
				    }
			    }						
		  volumeList1[0].url = img_t1
		  volumeList1[1].url = img_mask;
		  nv1.loadVolumes(volumeList1)
		  nv1.updateGLVolume()
	}
	imgEl.appendChild(btn)
	}	
	

	
</script>

  <!--var maskToShow = document.getElementById("mask_to_show")
  maskToShow.onchange = function() {
    switch(document.getElementById("mask_to_show").value) {		case "LOD-Brain":
			let root = './results/';
			let img_mask = root + imgs[i] + '_pred.nii.gz';
			volumeList1[1].url = img_mask;
			break
		case "FreeSurfer":
			let root = './results/';
			let img_mask = root + imgs[i] + '_GT.nii.gz';			volumeList1[1].url = img_mask;
			break
	    }
		nv1.loadVolumes(volumeList1);
		nv1.updateGLVolume();	
	  }-->

