#region License
/*
 * LogData.cs
 *
 * The MIT License
 *
 * Copyright (c) 2013-2014 sta.blockhead
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#endregion

using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace WebSocketSharp
{
  /// <summary>
  /// Represents a log data used by the <see cref="Logger"/> class.
  /// </summary>
  public class LogData
  {
    #region Private Fields

    private StackFrame _caller;
    private DateTime   _date;
    private LogLevel   _level;
    private string     _message;

    #endregion

    #region Internal Constructors

    internal LogData (LogLevel level, StackFrame caller, string message)
    {
      _level = level;
      _caller = caller;
      _message = message ?? String.Empty;
      // Logging can run on WebSocket worker threads. UtcNow avoids a native
      // local-timezone lookup that crashes Unity's bundled Mono globalization
      // code in Linux players when the bridge is unavailable during shutdown.
      _date = DateTime.UtcNow;
    }

    #endregion

    #region Public Properties

    public StackFrame Caller {
      get { return _caller; }
    }

    public DateTime Date {
      get { return _date; }
    }

    public LogLevel Level {
      get { return _level; }
    }

    public string Message {
      get { return _message; }
    }

    #endregion

    #region Public Methods

    public override string ToString ()
    {
      var timestamp = _date.ToString (
        "yyyy-MM-dd HH:mm:ss.fff 'UTC'", CultureInfo.InvariantCulture);
      var header = String.Format (
        CultureInfo.InvariantCulture, "{0}|{1,-5}|", timestamp, _level);
      var method = _caller.GetMethod ();
      var type = method.DeclaringType;
#if DEBUG
      var lineNum = _caller.GetFileLineNumber ();
      var headerAndCaller = String.Format (
        "{0}{1}.{2}:{3}|", header, type.Name, method.Name, lineNum);
#else
      var headerAndCaller = String.Format ("{0}{1}.{2}|", header, type.Name, method.Name);
#endif

      var messages = _message.Replace ("\r\n", "\n").TrimEnd ('\n').Split ('\n');
      if (messages.Length <= 1)
        return String.Format ("{0}{1}", headerAndCaller, _message);

      var log = new StringBuilder (
        String.Format ("{0}{1}\n", headerAndCaller, messages [0]), 64);

      var space = header.Length;
      var format = String.Format ("{{0,{0}}}{{1}}\n", space);
      for (var i = 1; i < messages.Length; i++)
        log.AppendFormat (format, "", messages [i]);

      log.Length--;
      return log.ToString ();
    }

    #endregion
  }
}
