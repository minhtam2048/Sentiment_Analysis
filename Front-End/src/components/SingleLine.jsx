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
      isLoading: false
    };
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(e) {
    this.setState({ content: e.target.value });
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
        `[{"content": "${this.state.content}", "resultInNumber": 0.1, "resultInBoolean": false}]`
      );

      // this.setState({
      //   dataToSend: data
      // });

      // console.log(this.state.dataToSend)

      let { data } = await httpCommon(dataToSend);
      console.log(data[0].resultInNumber);

      this.setState({
        dataRespond: data,
        trustNumber: data[0].resultInNumber,
        predict: data[0].resultInBoolean,
        isLoading: false
      });
    }
    // console.log(this.state.dataRespond);
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
