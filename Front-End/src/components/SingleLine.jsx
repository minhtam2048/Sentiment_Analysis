import React, { Component } from "react";
import { Input, Button } from "antd";
import httpCommon from "../DataService/common-http";
import { message } from "antd";
import "antd/dist/antd.css";
import "./simple.scss";
import EmojiHaHa from "./EmojiHaHa";
import EmojiSad from "./EmojiSad";
import Spinner from "./Spinner";

class SingleLine extends Component {
  constructor(props) {
    super(props);
    this.state = {
      editing: false,
      dataToSend: null,
      dataRespond: null,
      content: "",
      trustNumber: 0.0,
      predict: null,
      isLoading: false,
      language: 'vietnamese'
    };
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleVietnameseSubmit = this.handleVietnameseSubmit.bind(this);
    this.handleJapaneseSubmit = this.handleJapaneseSubmit.bind(this);
  }

  handleChange(e) {
    this.setState({ content: e.target.value });
  }

  handleVietnameseSubmit() {
    this.setState({ language: 'vietnamese'});
    console.log(this.state.language)
  }

  handleJapaneseSubmit() {
    this.setState({ language: 'japanese'});
    console.log(this.state.language)
  }

  async handleSubmit(e) {
    e.preventDefault();
    var dataToSend = null;
    console.log(this.state.content);

    if (this.state.content.trim() === "") {
      message.error("please input some texts");
    } else {
      this.setState({isLoading: true })
      dataToSend = JSON.parse(
        `{"language": "${this.state.language}","posts": [{"content": "${this.state.content}", "resultInNumber": 0.1, "resultInBoolean": false}]}`
      );

     

      let { data } = await httpCommon(dataToSend);

      this.setState({
        dataRespond: data,
        trustNumber: data.posts[0].resultInNumber,
        predict: data.posts[0].resultInBoolean,
        isLoading: false
      });
    }
  }

  render() {
    return (
      <React.Fragment>
        <div className="search-wrapper">
          <Input
            placeholder="Input text here"
            className="search-box"
            value={this.state.content}
            onChange={this.handleChange}
          />
        </div>
        <Button
          type="primary"
          className="custom-button"
          onClick={this.handleSubmit}
        >
          Predict
        </Button>
        <div>
          <Button className="custom-button" onClick={this.handleVietnameseSubmit}>vi</Button>
          <Button className="custom-button" onClick={this.handleJapaneseSubmit}>jp</Button>
        </div>
        {
          this.state.predict === null ?
          <div className="arrange-items">
            <EmojiHaHa/>
              {
              this.state.isLoading === true ? <span className="arrange-items"><Spinner/></span> :
              <React.Fragment>
              <span className="arrange-items">{this.state.trustNumber * 100}%</span>
              </React.Fragment>
              }
            <EmojiSad/>
              <div>reliability</div>
          </div> :
          this.state.predict === true ? 
          <div className="arrange-items">
            <EmojiHaHa/>
              {
                this.state.isLoading === true ? <span className="arrange-items"><Spinner/></span> :
              <span className="arrange-items">{this.state.trustNumber * 100}%</span>
              }
            <EmojiHaHa/>
            <div>reliability</div>
          </div> :
          <div className="arrange-items">
          <EmojiSad/>
            {
              this.state.isLoading === true ? <span className="arrange-items"><Spinner/></span> :
            <span className="arrange-items">{this.state.trustNumber * 100}%</span>
            }
          <EmojiSad/>
          <div>Reliability</div>
          </div>
        }
        
        {/* <div className="arrange-items">
          
          <EmojiHaHa/>
          
          <span className="arrange-items">{this.state.predict}%</span>

          <EmojiSad/>
        </div> */}
      </React.Fragment>
    );
  }
}

export default SingleLine;
