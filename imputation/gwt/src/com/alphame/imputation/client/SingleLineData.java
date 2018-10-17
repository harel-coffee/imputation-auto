package com.alphame.imputation.client;
import com.google.gwt.core.client.JavaScriptObject;

public class SingleLineData extends JavaScriptObject {                              // (1)
  // Overlay types always have protected, zero argument constructors.
  protected SingleLineData() {}                                              // (2)

  // JSNI methods to get stock data.
  public final native String get_content() /*-{ return this.content; }-*/; // (3)
  }


