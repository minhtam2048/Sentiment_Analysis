import React, { Component } from 'react';
import './App.css';
import DropFile from './components/DropFile';
import {BrowserRouter as Router, Route} from 'react-router-dom'
import SingleLine from './components/SingleLine';



class App extends Component {

  
  render() {
    return (
      <Router>
      <div className='App'>
        {/* <DropFile/> */}
        <Route path="/" exact component={DropFile}/>
        <Route path="/singleLine" exact component={SingleLine}/>
      </div>
      </Router>
    );
  }
}

export default App;
