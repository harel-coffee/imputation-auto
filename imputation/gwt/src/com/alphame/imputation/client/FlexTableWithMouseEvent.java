package com.alphame.imputation.client;
import com.google.gwt.event.dom.client.HasMouseMoveHandlers;
import com.google.gwt.event.dom.client.HasMouseOutHandlers;
import com.google.gwt.event.dom.client.HasMouseOverHandlers;
import com.google.gwt.event.dom.client.HasMouseUpHandlers;
import com.google.gwt.event.dom.client.HasDoubleClickHandlers;
import com.google.gwt.event.dom.client.MouseMoveEvent;
import com.google.gwt.event.dom.client.MouseMoveHandler;
import com.google.gwt.event.dom.client.MouseOutEvent;
import com.google.gwt.event.dom.client.MouseOutHandler;
import com.google.gwt.event.dom.client.MouseOverEvent;
import com.google.gwt.event.dom.client.MouseOverHandler;
import com.google.gwt.event.dom.client.MouseUpEvent;
import com.google.gwt.event.dom.client.MouseUpHandler;
import com.google.gwt.event.dom.client.DoubleClickEvent;
import com.google.gwt.event.dom.client.DoubleClickHandler;


import com.google.gwt.event.shared.HandlerRegistration;
import com.google.gwt.user.client.DOM;
import com.google.gwt.dom.client.Element;
import com.google.gwt.user.client.Event;
import com.google.gwt.user.client.ui.FlexTable;

public  class FlexTableWithMouseEvent extends FlexTable implements HasMouseOverHandlers, HasMouseOutHandlers ,HasMouseMoveHandlers , HasMouseUpHandlers,HasDoubleClickHandlers{
    /*
	private Element head;  
    private Element headerTr;
    */  
	public FlexTableWithMouseEvent(){
		super();    
		sinkEvents(Event.ONMOUSEOVER |Event.ONMOUSEOUT |Event.ONMOUSEUP |Event.ONDBLCLICK);  
    }
	
	@Override
	public HandlerRegistration addMouseOutHandler(MouseOutHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, MouseOutEvent.getType());
	}
	
	@Override
	public HandlerRegistration addMouseMoveHandler(MouseMoveHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, MouseMoveEvent.getType());
	}
	
	@Override
	public HandlerRegistration addMouseOverHandler(MouseOverHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, MouseOverEvent.getType());
	}
	
	@Override
	public HandlerRegistration addMouseUpHandler(MouseUpHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, MouseUpEvent.getType());
	}
	
	@Override
	public HandlerRegistration addDoubleClickHandler(DoubleClickHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, DoubleClickEvent.getType());
	}
	
	public Element getRowForMouseMoveEvent(MouseMoveEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }
	    Element tr = DOM.getParent(td);
	    if (tr == null){
		      return null;
	    }	    	
	    return tr;
	}
	
	public Element getRowForMouseOverEvent(MouseOverEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }
	    Element tr = DOM.getParent(td);
	    if (tr == null){
		      return null;
	    }	    	
	    return tr;
	}
	
	public Element getRowForMouseOutEvent(MouseOutEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }
	    Element tr = DOM.getParent(td);
	    if (tr == null){
		      return null;
	    }	    	
	    return tr;
	}
	
	public Element getCellForMouseMoveEvent(MouseMoveEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }    	
	    return td;
	}
	
	public Element getCellForMouseOverEvent(MouseOverEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }	
	    return td;
	}
	
	public Element getCellForMouseOutEvent(MouseOutEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }    	
	    return td;
	}
	
	public Element getCellForMouseUpEvent(MouseUpEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }    	
	    return td;
	}
}	


 