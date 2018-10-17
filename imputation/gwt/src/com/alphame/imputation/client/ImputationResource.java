package com.alphame.imputation.client;

import com.google.gwt.resources.client.ClientBundle;
import com.google.gwt.resources.client.ImageResource;
import com.google.gwt.resources.client.TextResource;

public interface ImputationResource extends ClientBundle{
	@Source("resource/logo.png")
	ImageResource logo();
	@Source("resource/upload.png")
	ImageResource upload();
	@Source("resource/view.png")
	ImageResource view();	
	@Source("resource/miscellaneous.png")
	ImageResource miscellaneous();
	@Source("resource/protein.png")
	ImageResource protein();
	@Source("resource/loading.gif")
	ImageResource loading();
	@Source("resource/imputation_legend.png")
	ImageResource imputation_legend(); 
	@Source("resource/polyphen_legend.png")
	ImageResource polyphen_legend(); 
	@Source("resource/sift_legend.png")
	ImageResource sift_legend(); 
	@Source("resource/asa_legend.png")
	ImageResource asa_legend(); 
	@Source("resource/gnomad_legend.png")
	ImageResource gnomad_legend(); 
	@Source("resource/provean_legend.png")
	ImageResource provean_legend(); 
	@Source("resource/question_mark.png")
	ImageResource question_mark(); 
	@Source("resource/error.png")
	ImageResource error(); 
	@Source("resource/upload_help.png")
	ImageResource upload_help(); 
	@Source("resource/view_help.jpg")
	ImageResource view_help(); 
	@Source("resource/help.png")	
	ImageResource help(); 
	@Source("resource/example.png")	
	ImageResource example(); 
	@Source("resource/imputation_help.html")
	TextResource help_html();
}
