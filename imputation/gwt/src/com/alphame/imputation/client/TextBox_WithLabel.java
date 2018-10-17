package com.alphame.imputation.client;
import com.google.gwt.user.client.ui.Composite;
import com.google.gwt.user.client.ui.TextBox;
import com.google.gwt.user.client.ui.Label;
import com.google.gwt.user.client.ui.HorizontalPanel;
import com.google.gwt.user.client.ui.Widget;

public  class TextBox_WithLabel extends Composite {
	private TextBox textbox = new TextBox();
	private Label label = new Label();
	public TextBox_WithLabel(String caption){
		HorizontalPanel panel = new HorizontalPanel();
		panel.add(label);
		panel.add(textbox);
		
		label.setText(caption);
    }
	
	public void setCaption(String caption) {
		label.setText(caption);
	}
	
	public void setText(String text) {
		textbox.setText(text);
	}	
}
	


 