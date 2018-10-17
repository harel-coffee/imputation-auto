package com.alphame.imputation.client;
import com.google.gwt.core.client.JavaScriptObject;

public class ErrorData extends JavaScriptObject {                              // (1)
  // Overlay types always have protected, zero argument constructors.
  protected ErrorData() {}                                              // (2)

  // JSNI methods to get stock data.
  public final native String get_error() /*-{ return this.error; }-*/; // (3)
  }


